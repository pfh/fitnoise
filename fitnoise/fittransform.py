"""

Optimal moderated log transformation.

"""

from __future__ import division

from .env import *
from . import distributions



def info(vec):
    n = vec.shape[0]
    residual = vec - vec.mean()
    residual2 = (residual*residual).sum()
    var = residual2 / n
    return -0.5*n*log(var)

def residual_info(residual):
    n = residual.shape[0]
    residual2 = (residual*residual).sum()
    var = residual2 / n
    return -0.5*n*log(var)

def vec_info(mat):
    #TODO: optimize
    n = mat.shape[0]
    residual = ( mat - mat.mean(axis=0)[None,:] )
    covar = (residual[:,:,None]*residual[:,None,:]).sum(axis=0) / n
    return -0.5*n*log(det(covar))

def spheroid_info(mat):
    residual = ( mat - mat.mean(axis=0)[None,:] ).flatten()
    residual2 = (residual*residual).sum()
    n = residual.shape[0]
    var = residual2 / n
    return -0.5*n*log(var)


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
        
        vparam = result._unpack_transform(vm, vpack)
        
        vy = result._apply_transform(vparam, vx)

        # This is just a fancy way of saying
        # minimize the correlation?
        # maximize the null component
        vcost = (
            spheroid_info(vy) #*(vq_null.shape[1]/vq_null.shape[0])
            #- vec_info(vy)
            - spheroid_info(dot(vy,vq_design)) #*(vq_null.shape[1]/vq_design.shape[1])
            - spheroid_info(dot(vy,vq_null))
            )
            
            
        
        vcost_grad = gradient.grad(vcost, vpack)
        
        func = theano.function(
            [vpack, vx, vq_design, vq_null],
            [vcost, vcost_grad],
            on_unused_input="ignore",
            )    
        
        def score(pack):
            result = func(pack,x,q_design,q_null)
            return result
        
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
        transform = result._unpack_transform(m, pack)
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


class Transform_polynomial(Transform):
    def __init__(self, order):
        assert order > 0
        self.order = order
    
    def _unpack_transform(self, m, pack):
        vecs = [ ]
        for i in xrange(self.order):
            vec, pack = pack[:m],pack[m:]
            vecs.append(vec)        
        return vecs
    
            
    def _apply_transform(self, param, x):
        poly = 1.0
        for i in xrange(self.order):
            poly = poly*x + param[i]
        return log(poly) * (1.0/(self.order*log(2.0)))
        
    
    def _configured(self):
        result = super(Transform_polynomial, self)._configured()
        m = result.x.shape[1]
        
        param_initial = (
             [ 0.0 ] * (self.order-1) * m +
             [ 1.0 ] * m 
             )
        
        param_bounds = (
            [(0.0, None)] * (self.order-1) * m +
            [(1e-10, None)] * m
            )
            
        assert len(param_initial) == len(param_bounds)
        return result._with(
            _param_initial = param_initial,
            _param_bounds = param_bounds,
            )


def transform(x, 
        design=None,
        order=2, 
        verbose=False):
    return Transform_polynomial(order).fit(x, design=design, verbose=verbose)



        