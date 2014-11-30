"""

Optimal moderated log transformation.

"""

from .env import *
from . import distributions



def info(vec):
    n = vec.shape[0]
    residual = vec - vec.mean()
    residual2 = (residual*residual).sum()
    var = residual2 / n
    return -0.5*n*log(var)
    #return -0.5*n*(
    #        log(2*numpy.pi)
    #        + log(var)             
    #        + 1.0
    #        )

def vec_info(mat):
    #TODO: optimize
    n = mat.shape[0]
    residual = ( mat - mat.mean(axis=0)[None,:] )
    covar = (residual[:,:,None]*residual[:,None,:]).sum(axis=0) / n
    return -0.5*n*log(det(covar))


def simple_vec_info(mat):
    #TODO: optimize
    residual = ( mat - mat.mean(axis=0)[None,:] ).flatten()
    residual2 = (residual*residual).sum()
    n = residual.shape[0]
    var = residual2 / n
    return -0.5*n*log(var)


def pull_out_triangle(vec, n):
    pieces = [ ]
    for i in xrange(1,n+1):
        pieces.append(vec[:i])
        pieces.append(zeros((n-i,)))
        vec = vec[i:]
    return concatenate(pieces).reshape((n,n)), vec

def pull_out_strict_triangle(vec, n):
    pieces = [ ]
    for i in xrange(0,n):
        pieces.append(vec[:i])
        pieces.append(zeros((n-i,)))
        vec = vec[i:]
    return concatenate(pieces).reshape((n,n)), vec

#print pull_out_triangle(numpy.array([1,2,3,4,5,6]), 3)


class Transform(Withable):
    def __repr__(self):
        result = self.__class__.__name__
        if hasattr(self, "y"):
            result += "\n"+self.optimization_result.message
            result += "\nTransform parameters: "+repr(self.transform)
        return result
    
    def _configured(self):
        return self
    
    def fit(self, x, design=None, verbose=False):
        x = as_matrix(x)
        n,m = x.shape
        
        if design is None:
           design = ones((m,1))
        design = as_matrix(design)
        
        q,r = qr_complete(design)
        q_design = q[:,:design.shape[1]]
        q_null = q[:,design.shape[1]:]
        
    
        result = self._with(
            x=x,
            design=design,
            null=q_null,
            )._configured()

        if verbose:
            print 'Initial guess:', result._param_initial
    
        initial = as_vector(result._param_initial)    
        bounds = result._param_bounds 
        
        vpack = tensor.dvector("pack")
        vx = tensor.dmatrix("x")        
        vq_design = tensor.dmatrix("vq_design")        
        vq_null = tensor.dmatrix("vq_null")        
        
        vm = vx.shape[1]
        
        vparam, vpack2 = result._unpack_transform(vm, vpack)
        #(vpack2 should be empty)
        
        #_vdelta = tensor.dscalar("delta")
        #_vy = result._apply_transform(vx+_vdelta, vparam)
        #vy = theano.clone(_vy, replace={_vdelta:0.0})
        #_vderiv = gradient.jacobian(_vy.flatten(), _vdelta)
        #vderiv = theano.clone(_vderiv, replace={_vdelta:0.0})
        
        #Unsure why this is so much faster
        #eps = 1e-4
        #choose eps scaled with x, for numerical precision
        #eps = ((vx+1)*1e-3)
        
        vy = result._apply_transform(vparam, vx)
        
        #vderiv = (result._apply_transform(vparam, vx+eps) - vy)/eps
        #
        #vtransform_cost = -log(vderiv).sum()
        ##vtransform_cost = -0.5 * log(dot(null.T,vderiv.T)**2).sum()  
        #
        #vdist_cost = -simple_vec_info(dot(vy,q_null))
        ##estimate additional cost of hidden dimensions ?!
        #vdist_cost = vdist_cost * (float(m) / (m-design.shape[1]))
        #
        #vcost = vtransform_cost + vdist_cost 
        

        vcost = -(
              simple_vec_info(dot(vy,vq_null))
            + vec_info(dot(vy,vq_design))
            - simple_vec_info(vy)
            )
        
        vcost_grad = gradient.grad(vcost, vpack)
        
        func = theano.function(
            [vpack, vx, vq_design, vq_null],
            [vcost, vcost_grad],
            #on_unused_input="ignore",
            )    
        
        def score(pack):
            result = func(pack,x,q_design,q_null)
            return result#[0]
        
        a = initial
        last = None
        for i in xrange(100):
            fit = scipy.optimize.minimize(
                score, 
                a,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                )    
            a = fit.x
            if verbose: 
                print 'Score:', fit.fun
            if last is not None and last-fit.fun <= 1e-6: break
            last = fit.fun
        
        if verbose:
            print 'Optimized:', list(fit.x)
            print fit.message
        
        pack = fit.x.copy()
        transform, pack = result._unpack_transform(m, pack)
        y = result._apply_transform(transform, x)
        y = y - y.mean(axis=0)
        
        adjustment = log(1e6 / x.sum(axis=0)).mean()/log(2.0)
        y_per_million = y + adjustment
        
        return result._with(
            optimization_result = fit,
            transform = transform,
            y = y,
            #y_per_million = y_per_million,
            )


class Transform_linear(Transform):
    def _unpack_transform(self, m, pack):
        vecs = [ ]
        for i in xrange(1):
            vec, pack = pack[:m],pack[m:]
            vecs.append(vec)
        
        return vecs, pack
    
            
    def _apply_transform(self, param, x):
        (b,) = param
        return log(x+b) * (1.0/log(2.0))
        
    
    def _configured(self):
        result = super(Transform_linear, self)._configured()
        m = result.x.shape[1]
        
        x_median = numpy.median(result.x,axis=0)
        guess = x_median+1
        
        param_initial = (
             list(guess) 
             )
        
        param_bounds = (
            [(1.0, None)]*m
            )
        assert len(param_initial) == len(param_bounds)
        return result._with(
            _param_initial = param_initial,
            _param_bounds = param_bounds,
            )


class Transform_quadratic(Transform):
    def _unpack_transform(self, m, pack):
        vecs = [ ]
        for i in xrange(2):
            vec, pack = pack[:m],pack[m:]
            vecs.append(vec)
        
        return vecs, pack
    
            
    def _apply_transform(self, param, x):
        b,c = param
        return log(x*x+b*x+c) * (0.5/log(2.0))
        
    
    def _configured(self):
        result = super(Transform_quadratic, self)._configured()
        m = result.x.shape[1]
        
        x_median = numpy.median(result.x,axis=0)
        guess = x_median+1
        
        param_initial = (
             list(guess) +
             list(guess**2) 
             )
        
        param_bounds = (
            [(0.0, None)]*m+
            [(1.0, None)]*m
            )
        assert len(param_initial) == len(param_bounds)
        return result._with(
            _param_initial = param_initial,
            _param_bounds = param_bounds,
            )


class Transform_cubic(Transform):
    def _unpack_transform(self, m, pack):
        vecs = [ ]
        for i in xrange(3):
            vec, pack = pack[:m],pack[m:]
            vecs.append(vec)
        
        return vecs, pack
    
            
    def _apply_transform(self, param, x):
        b,c,d = param
        return log(x*x*x+b*x*x+c*x+d) * (1.0/(3*log(2.0)))
        
    
    def _configured(self):
        result = super(Transform_cubic, self)._configured()
        m = result.x.shape[1]
        
        x_median = numpy.median(result.x,axis=0)
        guess = x_median+1
        
        param_initial = (
             list(guess) +
             list(guess**2) +
             list(guess**3)
             )
        
        param_bounds = (
            [(0.0, None)]*m+
            [(0.0, None)]*m+
            [(1.0, None)]*m
            )
        assert len(param_initial) == len(param_bounds)
        return result._with(
            _param_initial = param_initial,
            _param_bounds = param_bounds,
            )


class Transform_varstab3(Transform):
    def _unpack_transform(self, m, pack):
        vecs = [ ]
        for i in xrange(2):
            vec, pack = pack[:m],pack[m:]
            vecs.append(vec)
        
        return vecs, pack
    
            
    def _apply_transform(self, param, x):
        b,c = param
        return log(0.5*((x*(x+b)+c)**0.5)+0.5*x+0.25*b) / log(2.0)
        
    
    def _configured(self):
        result = super(Transform_varstab3, self)._configured()
        m = result.x.shape[1]
        
        x_median = numpy.median(result.x,axis=0)
        guess_b = x_median+1
        
        param_initial = (
             list(guess_b) +
             [0.0]*m 
             )
        
        param_bounds = (
            [(1e-4, None)]*m+
            [(0.0, None)]*m
            )
        assert len(param_initial) == len(param_bounds)
        return result._with(
            _param_initial = param_initial,
            _param_bounds = param_bounds,
            )


class Transform_varstab2(Transform):
    def _unpack_transform(self, m, pack):
        vecs = [ ]
        for i in xrange(1):
            vec, pack = pack[:m],pack[m:]
            vecs.append(vec)
        
        return vecs, pack
    
            
    def _apply_transform(self, param, x):
        [b] = param
        return log(0.5*((x*(x+b))**0.5)+0.5*x+0.25*b) / log(2.0)
        
    
    def _configured(self):
        result = super(Transform_varstab2, self)._configured()
        m = result.x.shape[1]
        
        x_median = numpy.median(result.x,axis=0)
        guess = x_median+1
        
        param_initial = (
             list(guess)
             )
        
        param_bounds = (
            [(1e-10, None)]*m
            )
        assert len(param_initial) == len(param_bounds)
        return result._with(
            _param_initial = param_initial,
            _param_bounds = param_bounds,
            )




class Transform_pow(Transform):
    def _unpack_transform(self, m, pack):
        vecs = [ ]
        for i in xrange(3):
            vec, pack = pack[:m],pack[m:]
            vecs.append(vec)
        
        return vecs, pack
    
            
    def _apply_transform(self, param, x):
        b,c,d = param
        
        arcsinh = lambda x: log(x+(x*x+1)**0.5)
        return arcsinh(b*(x**c)+d)/c*(1.0/log(2.0))
        #return log(b+x**c)/c * (1.0/log(2.0))
        
    
    def _configured(self):
        result = super(Transform_pow, self)._configured()
        m = result.x.shape[1]
        
        #x_median = numpy.median(result.x,axis=0)
        #guess = x_median+1
        
        param_initial = (
             [ 5.0 ] * m +
             [ 0.5 ] * m +
             [ 0.0 ] * m
             )
        
        param_bounds = (
            [(0.001, None)]*m+
            [(0.001, None)]*m+
            [(None, None)]*m
            )
        assert len(param_initial) == len(param_bounds)
        return result._with(
            _param_initial = param_initial,
            _param_bounds = param_bounds,
            )


class Transform_arcsinh(Transform):
    def _unpack_transform(self, m, pack):
        vecs = [ ]
        for i in xrange(2):
            vec, pack = pack[:m],pack[m:]
            vecs.append(vec)
        
        return vecs, pack
    
            
    def _apply_transform(self, param, x):
        b,c = param
        
        arcsinh = lambda x: log(x+(x*x+1)**0.5)
        return arcsinh(b*x+c)*(1.0/log(2.0))
        #return log(b+x**c)/c * (1.0/log(2.0))
        
    
    def _configured(self):
        result = super(Transform_arcsinh, self)._configured()
        m = result.x.shape[1]
        
        #x_median = numpy.median(result.x,axis=0)
        #guess = x_median+1
        
        param_initial = (
             [ 1.0 ] * m +
             [ 5.0 ] * m
             )
        
        param_bounds = (
            [(0.001, None)]*m+
            [(None, None)]*m
            )
        assert len(param_initial) == len(param_bounds)
        return result._with(
            _param_initial = param_initial,
            _param_bounds = param_bounds,
            )





TRANSFORMS = { 
    "linear"    : Transform_linear,
    "quadratic" : Transform_quadratic,
    "cubic"     : Transform_cubic,
    "varstab3"  : Transform_varstab3,
    "varstab2"  : Transform_varstab2,
    "pow"       : Transform_pow,
    "arcsinh"   : Transform_arcsinh,
    }


def transform(x, 
        design=None,
        transform="varstab2", 
        verbose=False):
    return TRANSFORMS[transform]().fit(x, design=design, verbose=verbose)



        