
"""

Two axes of time:
- execution sequence
- specialization sequence
  source -> bytecode -> ( theano expression )? -> value

A = LL'



x'Aix 
= x'(LL')ix 
= x'L'i Lix
= (Li x)' Lix

check L'i = Li' ... yes

"""

import warnings, copy

have_theano = False
try:
    import theano
    from theano import tensor
    from theano import gradient
    try:
        from theano.tensor import nlinalg, slinalg
        have_theano = True
    except:
        warnings.warn("Couln't import linear algebra for theano, need a more recent version")
except:
    pass


import numpy, numpy.linalg, numpy.random
import scipy, scipy.optimize, scipy.stats, scipy.special



def is_theanic(x):
    return have_theano and isinstance(x, tensor._tensor_py_operators)

def as_tensor(x, dtype='float64'):
    if not isinstance(x, numpy.ndarray) and \
       not is_theanic(x):
        x = numpy.array(x)            

    if not x.dtype == dtype:
        x = x.astype(dtype)

    return x

    
def as_scalar(x, dtype='float64'):
    x = as_tensor(x, dtype)
    assert x.ndim == 0
    return x


def as_vector(x, dtype='float64'):
    x = as_tensor(x, dtype)
    assert x.ndim == 1
    return x


def as_matrix(x, dtype='float64'):
    x = as_tensor(x, dtype)
    assert x.ndim == 2
    return x


def dot(a,b):
    a = as_tensor(a)
    b = as_tensor(b)
    return (
        tensor.dot if is_theanic(a) or is_theanic(b) 
        else numpy.dot
        )(a,b)


def take(a,indices,axis):
    if is_theanic(a) or is_theanic(indices):
        return tensor.take(a,indices,axis)
    else:
        return numpy.take(a,indices,axis)


def take2(a,i0,i1):
    return take(take(a,i0,0),i1,1)


def diag(x):
    x = as_tensor(x)
    return tensor.diag(x) if is_theanic(x) else numpy.diag(x)


def log(x):
    x = as_tensor(x)
    return tensor.log(x) if is_theanic(x) else numpy.log(x)


def exp(x):
    x = as_tensor(x)
    return tensor.exp(x) if is_theanic(x) else numpy.exp(x)


#From Numerical Recipes in C
def lanczos_gammaln(x):
    x = as_tensor(x)
    y = x
    tmp = x + 5.5
    tmp = tmp - (x+0.5)*log(tmp)
    ser = 1.000000000190015
    for cof in (
        76.18009172947146, 
        -86.50532032941677, 
        24.01409824083091, 
        -1.231739572450155, 
        0.1208650973866179e-2, 
        -0.5395239384953e-5,
        ):
        y = y + 1
        ser = ser + cof / y
    return -tmp + log(2.5066282746310005*ser/x)


def gammaln(x):
    x = as_tensor(x)
    return lanczos_gammaln(x) if is_theanic(x) else scipy.special.gammaln(x)


#for i in xrange(1,10):
#    x = i
#    print numpy.exp(gammaln(x)), numpy.exp(lanczos_gammaln(x))
#import sys;sys.exit(0)


def inverse(A):
    A = as_matrix(A)
    return (nlinalg.matrix_inverse if is_theanic(A) else numpy.linalg.inv)(A)


def det(A):
    A = as_matrix(A)
    return (nlinalg.det if is_theanic(A) else numpy.linalg.det)(A)

#Doesn't have gradient
def cholesky(A):
    A = as_matrix(A)
    return (slinalg.cholesky if is_theanic(A) else numpy.linalg.cholesky)(A)


#Doesn't have gradient
def qr_complete(A):
    A = as_matrix(A)
    if is_theanic(A):
        return nlinalg.QRFull('complete')(A)
    else:
        return numpy.linalg.qr(A, mode="complete")



# Agnostic as to whether numpy or theano
# Mvnormal(dvector('mean'), dmatrix('covar'))
#
# TODO: maybe store covar as cholesky decomposition
#
class Mvnormal(object):
    def __init__(self, mean, covar):
        self.mean = as_vector(mean)
        self.covar = as_matrix(covar)


    # Not available for theano
    @property
    def good(self):
        result = numpy.isfinite(self.mean)
        for i in xrange(len(self.covar)):
            result = result & numpy.isfinite(self.covar[i])
        return result
    
    
    def log_density(self, x):
        x = as_vector(x)
        offset = x - self.mean
        n = self.mean.shape[0]
        
        #TODO: cleverer linear algebra
        return -0.5*(
            log(2*numpy.pi)*n
            + log(det(self.covar))
            + dot(offset, dot(inverse(self.covar), offset))
            )

    
    def p_value(self, x):
        x = as_vector(x)
        offset = x - self.mean
        df = self.covar.shape[0]
        q = dot(offset, dot(inverse(self.covar), offset))
        return stats.chi2.sf(q, df=df)
    
    
    # Not available for theano
    def random(self):
        A = cholesky(self.covar)
        return self.mean + dot(A.T, numpy.random.normal(size=len(self.mean)))
        

    def transformed(self, A):
        A = as_matrix(A)
        return Mvnormal(
            dot(A,self.mean),
            dot(dot(A,self.covar),A.T)
            )

            
    def shifted(self, x):
        x = as_matrix(x)
        return Mvnormal(self.mean+x, self.covar)


    def marginal(self, i):
        i = as_vector(i, 'int32')
        return Mvnormal(take(self.mean,i,0), take2(self.covar,i,i))


    def conditional(self, i1,i2,x2):
        i1 = as_vector(i, 'int32')
        i2 = as_vector(i, 'int32')
        x2 = as_vector(x2)
        
        mean1 = take(self.mean,i1,0)
        mean2 = take(self.mean,i2,0)
        offset2 = x2-mean2
        
        covar11 = take2(self.covar,i1,i1) 
        covar12 = take2(self.covar,i1,i2)
        covar21 = take2(self.covar,i2,i1)
        covar22 = take2(self.covar,i2,i2)
        covar22inv = inverse(covar22)
        covar12xcovar22inv = dot(covar12, covar22inv)
        
        return Mvnormal(
            mean1 + dot(covar12xcovar22inv,offset2),
            covar11 - dot(covar12xcovar22inv,covar21)
            )


class Mvt(object):
    def __init__(self, mean, covar, df):
        self.mean = as_vector(mean)
        self.covar = as_matrix(covar)
        self.df = as_scalar(df)


    @property
    def good(self):
        result = numpy.isfinite(self.mean)
        for i in xrange(len(self.covar)):
            result = result & numpy.isfinite(self.covar[i])
        return result


    def log_density(self, x):
        x = as_vector(x)
        offset = x - self.mean
        p = self.covar.shape[0]
        v = self.df
        return (
            gammaln(0.5*(v+p))
            - gammaln(0.5*v)
            - (0.5*p)*log(numpy.pi*v)
            - 0.5*log(det(self.covar))
            - (0.5*(v+p))*log(1+dot(offset, dot(inverse(self.covar), offset))/v)
            )
    
    
    def p_value(self, x):
        x = as_vector(x)
        offset = x - self.mean
        p = self.covar.shape[0]
        q = dot(offset, dot(inverse(self.covar), offset)) / p
        return scipy.stats.f.sf(q, dfn=p, dfd=self.df)


    def random(self):
        A = cholesky(self.covar)
        return (
            self.mean 
            + dot(A.T,numpy.random.normal(size=len(self.mean)))
              * numpy.sqrt(self.df / numpy.random.chisquare(self.df)) 
            )
    
    
    def transformed(self, A):
        A = as_matrix(A)
        return Mvt(
            dot(A,self.mean),
            dot(dot(A,self.covar),A.T),
            self.df
            )

            
    def shifted(self, x):
        x = as_matrix(x)
        return Mvt(self.mean+x, self.covar, self.df)


    def marginal(self, i):
        i = as_vector(i, 'int32')
        return Mvt(take(self.mean,i,0), take2(self.covar,i,i), self.df)


    def conditional(self, i1,i2,x2):
        i1 = as_vector(i, 'int32')
        i2 = as_vector(i, 'int32')
        x2 = as_vector(x2)
        p2 = len(i2)
        
        mean1 = take(self.mean,i1,0)
        mean2 = take(self.mean,i2,0)
        offset2 = x2-mean2
        
        covar11 = take2(self.covar,i1,i1) 
        covar12 = take2(self.covar,i1,i2)
        covar21 = take2(self.covar,i2,i1)
        covar22 = take2(self.covar,i2,i2)
        covar22inv = inverse(covar22)
        covar12xcovar22inv = dot(covar12, covar22inv)
        
        df = self.df
        
        return Mvt(
            mean1 + dot(covar12xcovar22inv,offset2),
            (covar11 - dot(covar12xcovar22inv,covar21))
              * ((df + dot(offset2,dot(covar22inv,offset2))) / (df + p2)),
            df + p2
            )






def _fit_noise(y, designs, get_dist, initial, 
               aux, bounds, use_theano, verbose):
    n,m = y.shape
    
    items = [ ]
    row_tQ2_z2 = { }
    for row in xrange(n):
        design = designs[row]
        retain = numpy.arange(m)[ 
            numpy.isfinite(y[row]) & get_dist(initial,aux[row]).good
            ]
        if len(retain) <= design.shape[1]: continue
        
        Q,R = qr_complete(design[retain])
        i2 = numpy.arange(design.shape[1],len(retain))        
        tQ2 = Q[:,i2].T
        z2 = dot(tQ2, y[row,retain])
        items.append( (aux[row], retain, tQ2, z2) )
        row_tQ2_z2[row] = (tQ2, z2)
    
    
    def score_row(param, aux, retain, tQ2, z2):
        return -(
            get_dist(param, aux)
            .marginal(retain)
            .transformed(tQ2)
            .log_density(z2)
            )
    
    
    if use_theano and not have_theano:
        warnings.warn("Couldn't import theano, calculations may be slow")
        use_theano = False
    
    # Non-theanic
    if not use_theano:
        def score(param):
            total_value = sum(
                score_row(param,*item)
                for item in items
                )
            if verbose:
                print param, total_value
            return total_value

        return scipy.optimize.minimize(
            score, initial, 
            method="L-BFGS-B",
            bounds=bounds), row_tQ2_z2

    vaux = tensor.ivector("aux")
    vretain = tensor.ivector("retain")
    vtQ2 = tensor.dmatrix("tQ2")
    vz2 = tensor.dvector("z2")
    vparam = tensor.dvector("param")

    vvalue = score_row(vparam, vaux, vretain, vtQ2, vz2)
    vgradient = gradient.grad(vvalue, vparam)
    
    svalue = theano.shared(numpy.zeros(()))
    sgradient = theano.shared(numpy.zeros(len(initial)))
    
    vg_func = theano.function(
        [vparam,vaux,vretain,vtQ2,vz2], 
        [],
        updates=[
            (svalue,svalue+vvalue),
            (sgradient,sgradient+vgradient),
            ],
        on_unused_input="ignore",
        allow_input_downcast=True)
    
    def value_gradient(items, param):
        svalue.set_value(0.0)
        sgradient.set_value(numpy.zeros(len(param)))
        for item in items:
            vg_func(param,*item)
        if verbose:
            print param, svalue.get_value()
        return svalue.get_value().copy(), sgradient.get_value().copy()
    
    score = lambda param: value_gradient(items, param) 
    
    #import IPython.parallel
    #c = IPython.parallel.Client()
    #view = c[:]
    #
    #view.execute("import numpy, theano")
    #view.scatter('items', items)
    #
    #view['thingy'] = dict([(name,locals()[name]) for name in
    #    ['initial','vaux','vretain','vtQ2',
    #     'vz2','vparam','vvalue','vgradient']])
    #
    #view.execute("""if 1:
    #for name in thingy:
    #    locals()[name] = thingy[name]
    #
    #svalue = theano.shared(numpy.zeros(()))
    #sgradient = theano.shared(numpy.zeros(len(initial)))
    #
    #vg_func = theano.function(
    #    [vparam,vaux,vretain,vtQ2,vz2], 
    #    #[vvalue,vgradient],
    #    [],
    #    updates=[
    #        (svalue,svalue+vvalue),
    #        (sgradient,sgradient+vgradient),
    #        ],
    #    on_unused_input='ignore',
    #    allow_input_downcast=True)
    #    """, block=True)
    #view.execute("""def value_gradient(items, param):
    #    svalue.set_value(0.0)
    #    sgradient.set_value(numpy.zeros(len(param)))
    #    for item in items:
    #        vg_func(param.copy(),*item)
    #    return svalue.get_value().copy(), sgradient.get_value().copy()
    #    """)
    #view.execute("""doit = lambda: value_gradient(items,param)""")
    #
    #def score(param):
    #    total_value = 0.0
    #    total_gradient = numpy.zeros(len(param))
    #    view['param'] = param
    #    for value, gradient in view.apply_sync(lambda: doit()):
    #        total_value = total_value + value
    #        total_gradient += gradient
    #    if verbose:
    #        print param, total_value
    #    return total_value, total_gradient 
    
    
    return scipy.optimize.minimize(
        score, initial,
        method="L-BFGS-B", 
        jac=True,
        bounds=bounds
        ), row_tQ2_z2
    
    #vvalue = score_row(vaux, vretain, vtQ2, vz2, vparam)
    #vgradient = gradient.grad(vvalue, vparam)
    #vhessian = gradient.hessian(vvalue, vparam)
    #
    #svalue = theano.shared(0.0)
    #sgradient = theano.shared(numpy.zeros(len(initial)))
    #shessian = theano.shared(numpy.zeros((len(initial),len(initial))))
    #
    #vgh_func = theano.function(
    #    [vaux,vretain,vtQ2,vz2,vparam], 
    #    [],
    #    updates = [
    #        (svalue,svalue+vvalue),
    #        (sgradient,sgradient+vgradient),
    #        (shessian,shessian+vhessian),
    #        ],
    #    on_unused_input='ignore',
    #    allow_input_downcast=True)
    #
    #def value_gradient_hessian(param):
    #    svalue.set_value(0.0)
    #    sgradient.set_value(numpy.zeros(len(param)))
    #    shessian.set_value(numpy.zeros((len(param),len(param))))
    #    for item in items:
    #        vgh_func(item.aux,item.retain,item.tQ2,item.z2,param)
    #    if verbose:
    #        print param, svalue.get_value()
    #    return svalue.get_value().copy(), sgradient.get_value().copy(), shessian.get_value().copy()
    #
    #cache = { }
    #def get(param):
    #    key = tuple(param)
    #    if key not in cache:
    #        cache[key] = value_gradient_hessian(key)
    #    return cache[key]
    #
    ##return scipy.optimize.minimize(
    ##    lambda x: get(x)[0], initial,
    ##    method='trust-ncg', jac=lambda x: get(x)[1], hess=lambda x: get(x)[2],
    ##    bounds=((0.0,100.0),)*len(initial)
    ##    )
    #
    #param = initial
    #for i in xrange(30):
    #    v,g,h = value_gradient_hessian(param)
    #    #print 'D', numpy.diag(h)
    #    #h += numpy.identity(h.shape[0]) * 1e-6
    #    #print 'D', numpy.diag(h)
    #    param = param - numpy.linalg.solve(h,g)
    #return param



class Dataset(object):
    def __init__(self, y, context={}):
        self.y = as_matrix(y)
        self.context = context

def as_dataset(dataset):
    if isinstance(dataset, Dataset):
        return dataset
    return Dataset(dataset)


class Model(object):
    def _with(self, **kwargs):
        result = copy.copy(self)
        for name in kwargs:
            setattr(result,name,kwargs[name])
        return result
    
    def __repr__(self):
        result = '%s\n' % self.__class__.__name__
        if hasattr(self,'param'):
            result += self._describe_noise(self.param)
            result += 'noise p-value = %f\n' % self.noise_combined_p_value
        return result
    
    def _describe_noise(self, param):
        return ''
    
    def fit_noise(self, data, 
            noise_design=None, 
            control_design=None, controls=None, use_theano=True,verbose=False):
        data = as_dataset(data)
        noise_design = as_matrix(noise_design)
        
        n,m = data.y.shape
        if noise_design is None: noise_design = numpy.ones((m,1))
        if control_design is None: control_design = numpy.ones((m,1))
        if controls is None: controls = [False]*n
        
        noise_design = as_matrix(noise_design)
        control_design = as_matrix(control_design)
        controls = as_vector(controls, 'bool')
        
        result = self._with(
            data=data,
            aux=numpy.zeros((data.y.shape[0],0)),
            noise_design=noise_design,
            control_design=control_design,
            controls=controls,
            )
        
        result = result._configured()
        
        designs = [ (noise_design if item else control_design) for item in controls ]

        fit, row_tQ2_z2 = _fit_noise(
            y=result.data.y, 
            designs=designs, 
            aux=result._aux,
            get_dist=result._get_dist,
            initial=result._initial,
            bounds=result._bounds,
            use_theano=use_theano,
            verbose=verbose,
            )
        param = fit.x
        
        noise_dists = [
            result._get_dist(param, result._aux[i])
            for i in xrange(n)
            ]
        
        noise_p_values = numpy.repeat(numpy.nan, n)
        for row,(tQ2,z2) in row_tQ2_z2.items():
            noise_p_values[row] = (
                noise_dists[row]
                .transformed(tQ2)
                .p_value(z2)
                )
        
        #Bonferroni combined p value
        good_p = noise_p_values[numpy.isfinite(noise_p_values)]
        if not len(good_p):
            noise_combined_p_value = 0.0
        else:
            noise_combined_p_value = min(1,numpy.minimum.reduce(good_p)*len(good_p))
        
        return result._with(
            optimization_result = fit,
            param = param,
            noise_dists = noise_dists,
            noise_p_values = noise_p_values,
            noise_combined_p_value = noise_combined_p_value,
            )


class Model_t_mixin(Model):
    """ This mix-in converts an Mv_normal model to an Mvt model. """

    def _describe_noise(self, param):
        result = super(Model_t_mixin,self)._describe_noise(param[1:])
        result += 'prior df = %f\n' % param[0]
        return result 

    def _get_dist(self, param, aux):
        inner_dist = super(Model_t_mixin, self)._get_dist(param[1:], aux)
        assert type(inner_dist) is Mvnormal
        return Mvt(inner_dist.mean, inner_dist.covar, param[0])

    def _configured(self):
        result = super(Model_t_mixin,self)._configured()
        return result._with(
            _initial=[5.0] + list(result._initial),
            _bounds=[(1e-3, 100.0)] + list(result._bounds),
            )


class Model_normal_standard(Model):
    def _describe_noise(self, param):
        return "s.d. = %f\n" % numpy.sqrt(param[0])
    
    def _get_dist(self, param, aux):
        m = self.data.y.shape[1]
        return Mvnormal(
            numpy.zeros(m),
            diag(param[0] * aux),
            )

    def _configured(self):
        if 'weights' in self.data.context:
            _aux = 1.0 / as_matrix(self.data.context['weights'])
        else:
            _aux = numpy.ones(self.data.y.shape)
        
        flat = (self.data.y / numpy.sqrt(_aux)).flat
        flat = flat[numpy.isfinite(flat)]
        var = numpy.var(flat) if len(flat) > 3 else 1.0
        
        return self._with(
            _aux=_aux,
            _initial = [var],
            _bounds = [(var*1e-6,var*2.0)],
            )


model_normal_standard = Model_normal_standard()


class Model_t_standard(Model_t_mixin, Model_normal_standard): pass
model_t_standard = Model_t_standard()






if __name__ == "__main__":
    numpy.random.seed(1)    
    dist = Mvt([5,5,5,5],numpy.identity(4), 5.0)
    data = numpy.array([ dist.random() for i in xrange(1000) ])
    print model_t_standard.fit_noise(data, [[1]]*4)




