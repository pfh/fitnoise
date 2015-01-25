
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

from .env import *
from .distributions import Mvnormal, Mvt
from . import p_adjust


def _fit_noise(y, designs, get_model_cost, get_dist, initial, 
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
    
    
    if len(initial) == 0:
       return Withable()._with(x=numpy.zeros(0)), row_tQ2_z2
    
    if use_theano and not have_theano:
        warnings.warn("Couldn't import theano, calculations may be slow")
        use_theano = False
    
    # Non-theanic
    if not use_theano:
        def score(param):
            total_value = get_model_cost(param) + \
                sum(
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

    vcost_value = get_model_cost(vparam)
    if not is_theanic(vcost_value):
        # It's a constant, disregard.
        cost_vg_func = lambda param: None
    else:
        vcost_gradient = gradient.grad(vcost_value, vparam)
        
        cost_vg_func = theano.function(
            [vparam], 
            [],
            updates=[
                (svalue,svalue+vcost_value),
                (sgradient,sgradient+vcost_gradient),
                ],
            on_unused_input="ignore",
            allow_input_downcast=True)
    
    def value_gradient(items, param):
        svalue.set_value(0.0)
        sgradient.set_value(numpy.zeros(len(param)))
        cost_vg_func(param)
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

    

class Model(Withable):
    """
    
        score = mean residual log2 density
    """
    
    def __init__(self):
        pass

    def __repr__(self):
        if hasattr(self,'param'):
            result = self._describe_noise(self.param)
            result += 'noise combined p-value = %f\n' % self.noise_combined_p_value
            result += 'noise fit score = %f bits\n' % self.score
        else:
            result = '<%s>\n' % self.__class__.__name__
        return result
    
    def _unpack(self, packed):
        return Withable()
    
    def _describe_noise(self, param):
        return ''
    
    def _model_cost(self, param):
        return 0.0
    
    def fit_noise(self, data, 
            design=None, 
            control_design=None, controls=None, use_theano=True,verbose=False):
        data = as_dataset(data)
        design = as_matrix(design)
        
        n,m = data.y.shape
        if design is None: design = numpy.ones((m,1))
        if control_design is None: control_design = numpy.ones((m,1))
        if controls is None: controls = [False]*n
        
        design = as_matrix(design)
        control_design = as_matrix(control_design)
        controls = as_vector(controls, 'bool')
        
        result = self._with(
            data=data,
            aux=numpy.zeros((data.y.shape[0],0)),
            noise_design=design,
            control_design=control_design,
            controls=controls,
            )
        
        result = result._configured()
        
        designs = [ (control_design if item else design) for item in controls ]

        fit, row_tQ2_z2 = _fit_noise(
            y=result.data.y, 
            designs=designs, 
            aux=result._aux,
            get_model_cost=lambda param: result._model_cost(result._unpack(param)),
            get_dist=lambda param, aux: result._get_dist(result._unpack(param),aux),
            initial=result._initial,
            bounds=result._bounds,
            use_theano=use_theano,
            verbose=verbose,
            )
        param = result._unpack(fit.x)
        
        if verbose:
            print "Model cost:", result._model_cost(param)
        
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
        
        n_residuals = sum(m - item.shape[1] for item in designs)
        
        return result._with(
            optimization_result = fit,
            param = param,
            score = fit.fun / n_residuals,
            noise_dists = noise_dists,
            noise_p_values = noise_p_values,
            noise_combined_p_value = noise_combined_p_value,
            )


    def fit_coef(self, design=None):
        if design is None:
            design = self.noise_design
        design = as_matrix(design)
    
        n,m = self.data.y.shape
        n_coef = design.shape[1]
        coef_dists = [ None ]*n
        
        coef = numpy.tile(numpy.nan, (n,n_coef))
        #TODO:
        #coef_lower95 = numpy.tile(numpy.nan, (n,m))
        #coef_upper95 = numpy.tile(numpy.nan, (n,m))
        
        for row in xrange(n):
            if not self.noise_dists[row]: continue

            retain = numpy.arange(m)[ 
                numpy.isfinite(self.data.y[row]) & self.noise_dists[row].good
                ]
            
            if len(retain) < n_coef: continue
            if numpy.linalg.matrix_rank(design[retain]) < n_coef: continue
            
            Q,R = qr_complete(design[retain])
            i1 = numpy.arange(n_coef)
            i2 = numpy.arange(n_coef, len(retain))
            Rinv = inverse(R[i1])
            Q2 = Q[:,i2]
            
            z = dot(Q.T, self.data.y[row])
            
            # Distribution of noise in z1, given z2
            cond_dist = (
                self.noise_dists[row]
                .marginal(retain)
                .transformed(Q.T)
                .conditional(i1, i2, z[i2])
                )
            
            # Distribution of  Rinv . (z1 - noise)
            coef_dists[row] = (
                cond_dist
                .shifted(-z[i1])
                .transformed(-Rinv)
                )
            
            coef[row] = coef_dists[row].mean
        
        return self._with(
            design = design,
            coef = coef,
            coef_dists = coef_dists,
            )
    
    def test(self, coef=None, contrasts=None):
        if coef:
            assert contrasts is None
            contrasts = numpy.zeros((self.design.shape[1], len(coef)))
            for i in xrange(len(coef)):
                contrasts[coef[i],i] = 1.0
        
        assert contrasts is not None
        contrasts = as_matrix(contrasts)
        
        n = self.data.y.shape[0]
        
        contrast_dists = [ None ]*n
        contrasted = numpy.tile(numpy.nan, (n,contrasts.shape[1]))
        p_values = numpy.tile(numpy.nan, n)
        
        for row in xrange(n):
            if not self.coef_dists[row]: continue
            
            contrast_dists[row] = self.coef_dists[row].transformed(contrasts.T)
            contrasted[row] = contrast_dists[row].mean
            p_values[row] = contrast_dists[row].p_value(numpy.zeros(contrasts.shape[1]))
        
        return self._with(
            contrasts=contrasted,
            contrast_dists=contrast_dists,
            p_values=p_values,
            q_values=p_adjust.fdr(p_values),
            )


class Model_t_mixin(Model):
    """ This mix-in converts an Mv_normal model to an Mvt model. """

    def _unpack(self, pack):
        df = pack[0]
        return (
            super(Model_t_mixin,self)
            ._unpack(pack[1:])
            ._with(df=df)
            )


    def _describe_noise(self, param):
        result = 'prior df = %f\n' % param.df
        result += super(Model_t_mixin,self)._describe_noise(param)
        return result 


    def _get_dist(self, param, aux):
        inner_dist = super(Model_t_mixin, self)._get_dist(param, aux)
        assert type(inner_dist) is Mvnormal
        return Mvt(inner_dist.mean, inner_dist.covar, param.df)


    def _configured(self):
        result = super(Model_t_mixin,self)._configured()
        return result._with(
            _initial=[5.0] + list(result._initial),
            _bounds=[(1e-3, 100.0)] + list(result._bounds),
            )


class Model_factors_mixin(Model):
    """ This mix-in adds unknown batch effect(s) to the model. """
    
    def __init__(self, n_factors, *etc):
        self.n_factors = n_factors
        super(Model_factors_mixin,self).__init__(*etc)

    
    def _unpack(self, pack):
        m2 = self._factor_Q2.shape[1]
        factors = [ ]
        for i in xrange(self.n_factors):
            factors.append(dot(self._factor_Q2, pack[:m2]))
            pack = pack[m2:]
        
        return (
            super(Model_factors_mixin,self)
            ._unpack(pack)
            ._with(factors=factors)
            )


    def _describe_noise(self, param):
        m2 = self._factor_Q2.shape[1]
        result = ''
        for i in xrange(self.n_factors):
            result += 'factor: %s\n' % ', '.join('%f'%item for item in param.factors[i])
        result += super(Model_factors_mixin,self)._describe_noise(param)
        return result

    
    def _model_cost(self, param):
        """
        It should always be possible to reduce this cost to zero.
        """
        cost = super(Model_factors_mixin,self)._model_cost(param)
        for i in xrange(self.n_factors):
            for j in xrange(i):
                cost = cost + dot(param.factors[i],param.factors[j]) ** 2
        return cost

    
    def _get_dist(self, param, aux):
        m, m2 = self._factor_Q2.shape
        covar = zeros((m,m))
        for i in xrange(self.n_factors):
            covar = covar + outer(param.factors[i],param.factors[i])
        dist = super(Model_factors_mixin, self)._get_dist(param, aux)
        return dist.plus_covar(covar)

    
    def _configured(self):
        result = super(Model_factors_mixin, self)._configured()
        m = self.data.y.shape[1]
        #flat = self.data.y.flat
        #flat = flat[numpy.isfinite(flat)]
        #var = numpy.var(flat) if len(flat) > 3 else 1.0
        #bound = numpy.sqrt(var)
        
        if numpy.any(self.controls):
            design = self.control_design
        else:
            design = self.noise_design
        Q,R = qr_complete(design)
        Q2 = Q[:,design.shape[1]:]
        m2 = Q2.shape[1]
        
        good = numpy.all(numpy.isfinite(self.data.y), axis=1)
        goody = dot(Q2.T, self.data.y[good].T).T
        u,s,v = numpy.linalg.svd(goody, full_matrices=0)
        
        rows = numpy.argsort(-numpy.abs(s))[:self.n_factors]
        initial = list((v[rows] * s[rows][:,None]).flatten() / numpy.sqrt(goody.shape[0]))

        return result._with(
             _factor_Q2 = Q2,
#             _initial=[0.0]*(m*self.n_factors) + result._initial,
             _initial=initial + result._initial,
             _bounds=[(None,None)]*(m2*self.n_factors) + result._bounds,
             )


class Model_normal(Model):
    def _unpack(self, pack):
        return Withable()._with(variance=pack[0])
            
    
    def _describe_noise(self, param):
        return "s.d. = %f\n" % numpy.sqrt(param.variance)
    
    
    def _get_dist(self, param, aux):
        m = aux.shape[0]
        return Mvnormal(
            zeros((m,)),
            diag(param.variance * aux),
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

class Model_t(Model_t_mixin, Model_normal): pass

class Model_normal_factors(
    Model_factors_mixin, Model_normal
    ): pass

class Model_t_factors(
    Model_t_mixin, Model_factors_mixin, Model_normal
    ): pass


class Model_t_independent(Model):
    def _describe_noise(self, param):
        return ""
    
    def _get_dist(self, param, aux):
        return Mvt(
            zeros((aux.shape[0],)),
            diag(aux),
            1e-10 #Approximately zero
            )

    def _configured(self):
        if "weights" in self.data.context:
            _aux = 1.0 / as_matrix(self.data.context["weights"])
        else:
            _aux = numpy.ones(self.data.y.shape)
        
        return self._with(
            _aux=_aux,
            _initial = [],
            _bounds = [],
            )

class Model_t_independent_factors(
    Model_factors_mixin, Model_t_independent
    ): pass



class Model_normal_patseq(Model):
    """ Differential tail length detection in PAT-Seq """

    def _unpack(self, pack):
        return Withable()._with(
            read_variance = pack[0],
            sample_variance = pack[1],
            )
    
    def _describe_noise(self, param):
        return "variance = %f^2 / reads + (%f * tail)^2" % (
            numpy.sqrt(param.read_variance), 
            numpy.sqrt(param.sample_variance)
            )
    
    def _get_dist(self, param, aux):
        m = aux.shape[0]//2
        count = aux[:m]
        tail = aux[m:]
        return Mvnormal(
            zeros((aux.shape[0],)),
            diag( param.read_variance/count + param.sample_variance*(tail*tail) )
            )

class Model_t_patseq(Model_t_mixin, Model_normal_patseq): pass





class Model_normal_patseq_v2(Model):
    """ Differential tail length detection in PAT-Seq, slight variation on model """

    def _unpack(self, pack):
        return Withable()._with(
            read_variance = pack[0],
            sample_variance = pack[1],
            )
    
    def _describe_noise(self, param):
        return "variance = (%f^2 / reads + %f^2) * tail^2" % (
            numpy.sqrt(param.read_variance), 
            numpy.sqrt(param.sample_variance)
            )
    
    def _get_dist(self, param, aux):
        m = aux.shape[0]//2
        count = aux[:m]
        tail = aux[m:]
        return Mvnormal(
            zeros((aux.shape[0],)),
            diag( (param.read_variance/count + param.sample_variance)*(tail*tail) )
            )

class Model_t_patseq_v2(Model_t_mixin, Model_normal_patseq_v2): pass


if __name__ == "__main__":
    numpy.random.seed(1)    
    dist = Mvt([5,5,5,5],numpy.identity(4), 5.0)
    data = numpy.array([ dist.random() for i in xrange(1000) ])
    print model_t_standard.fit_noise(data, [[1]]*4)




