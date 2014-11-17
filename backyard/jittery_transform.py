"""

Optimal moderated log transformation.

(x+a)^2 + b = x^2+x
xx+2ax+aa+bb = xx+x
2ax+aa+bb = x
a = 0.5
b = -0.5

"""

from .env import *
from . import distributions


def pull_out_triangle(vec, n):
    pieces = [ ]
    for i in xrange(1,n+1):
        pieces.append(vec[:i])
        pieces.append(zeros((n-i,)))
        vec = vec[i:]
    return concatenate(pieces).reshape((n,n)), vec

#print pull_out_triangle(numpy.array([1,2,3,4,5,6]), 3)


class Transform(Withable):
    order = 3
    
    def _unpack_distribution(self, m, pack):
        #TODO: allow m to be theanic
        m = self.x.shape[1]

        df, pack = pack[0], pack[1:]
        #mean, pack = pack[:m], pack[m:]
        L, pack = pull_out_triangle(pack, m)
        covar = dot(L,L.T)
        return distributions.Mvt(zeros((m,)), covar, df), pack
    
    
    def _unpack_transform(self, m, pack):
        vecs = [ ]
        for i in xrange(self.order):
            vec, pack = pack[:m],pack[m:]
            vecs.append(vec)
        return vecs, pack
        
        #a,b,c,pack = pack[:m],pack[m:m*2],pack[m*2:m*3],pack[m*3:]
        
        #a = exp(a)
        #b = exp(b)
        #c = exp(c)

        #x = self.precision * -0.5
        
        #x = -0.5*self.precision
        
        #Gradient must be positive        
        #b = b - 2*a*x
        
        #Value must be positive
        #base = a*x*x+b*x
        #c = c - base # base.min(axis=0)

        #return Withable()._with(a=a,b=b,c=c), pack
    
            
    def _apply_transform(self, x, param):
        #a = param.a
        #b = param.b
        #c = param.c
        # Integral 1/sqrt( x*x+a*x+b ) 
        # + a constant so it behaves like log in the limit        
        
        #sqrt(aa+b)+0.5*a = c
        #c-a/2 = sqrt(aa+b)
        # (c-a/2)^2 = aa+b
        #b = (c-a/2)^2 - aa
        
        #b such that all have same value at zero
                
        
        x = x+self.precision*0.5
        a,b,c = param
        
        xm = self.x_median        
        a = a - log(0.5*((xm*(xm+b)+c)**0.5)+0.5*xm+0.25*b) / log(2.0)
        
        return a + log(0.5*((x*(x+b)+c)**0.5)+0.5*x+0.25*b) / log(2.0)
        
        #a = a - log(self.x_median**2+b*self.x_median+c)*(0.5/log(2.0))
        #return a + log(x*x+b*x+c)*(0.5/log(2.0))
        
        #y = 1.0
        #for i in xrange(1,self.order):
        #    y = y * x + param[i]
            # + param[i] * concatenate([x,i+zeros(x.shape)]).max(axis=0)
        #return param[0] + ((y)**param[1])*(1.0/self.order/log(2.0))
        
    
    def _configured(self):
        m = self.x.shape[1]
        param_initial = [0.0]*m + [1.0]*m + [1.0]*m*(self.order-2)
        
        param_bounds = [(None,None)]*m + [(0.01, None)]*m + [(0.0, None)]*m*(self.order-2)
        
        assert len(param_initial) == len(param_bounds)
        
        
        dist_initial = [5.0] #+ [0.0]*m
        for i in xrange(m):
            dist_initial.extend([1.0]*i+[2.0])
        dist_bounds = [(1.0,1000.0)] + [(None,None)] * (len(dist_initial)-1)
        return self._with(
            _param_initial = param_initial,
            _param_bounds = param_bounds,
            _dist_initial = dist_initial,
            _dist_bounds = dist_bounds,
            )
    
    def fit(self, x, precision=1.0, verbose=False, 
            random_seed=12345, min_n=10000):
        x = as_matrix(x)
        n,m = x.shape
        
        x_median = numpy.median(x,axis=0)
    
        result = self._with(
            x=x,
            x_median=x_median,
            precision=precision
            )._configured()
    
        initial = numpy.concatenate([result._param_initial,result._dist_initial])    
        bounds = result._param_bounds + result._dist_bounds
        
        rng = numpy.random.RandomState(random_seed)
        
        reps = 1
        #while precision != 0.0 and n and n*reps < min_n: reps += 1
        x_jittered = numpy.tile(x, (reps,1))
        x_jittered = x_jittered + (0.5-rng.random_sample(size=x_jittered.shape))*precision

        vpack = tensor.dvector("pack")
        vx = tensor.dmatrix("x")        
        
        vm = vx.shape[1]
        
        vparam, vpack2 = result._unpack_transform(vm, vpack)
        vdist, vpack3 = result._unpack_distribution(vm, vpack2)
        #(vpack3 should be empty)
        
        #_vdelta = tensor.dscalar("delta")
        #_vy = result._apply_transform(vx+_vdelta, vparam)
        #vy = theano.clone(_vy, replace={_vdelta:0.0})
        #_vderiv = gradient.jacobian(_vy.flatten(), _vdelta)
        #vderiv = theano.clone(_vderiv, replace={_vdelta:0.0})
        
        #Unsure why this is so much faster
        eps = 1e-3
        vy = result._apply_transform(vx, vparam)
        vderiv = (result._apply_transform(vx+eps, vparam) - vy)/eps
        
        vtransform_cost = -log(vderiv).sum()  #???
        
        vdist_cost = -vdist.log_density(vy).sum()
        
        vcost = vtransform_cost + vdist_cost 
        vcost_grad = gradient.grad(vcost, vpack)
        
        func = theano.function(
            [vx, vpack],
            [vcost, vcost_grad],
            )    
        
        def score(pack):
            old = pack
            pack = pack.copy()
            for i,(a,b) in enumerate(bounds):
                if a is not None: pack[i] = max(a,pack[i])
                if b is not None: pack[i] = min(b,pack[i])
            err = ((old-pack)**2).sum()
            
            result = func(x_jittered,pack)
            if verbose:
                print pack, result[0]
            return result[0]+err, result[1]
        
        a = initial
        for i in xrange(1):
            fit = scipy.optimize.minimize(
                score, 
                a,
                #method="Nelder-Mead",
                method="L-BFGS-B", 
                #method="SLSQP",
                #method="BFGS",
                jac=True,
                bounds=bounds,
                #options=dict(ftol=1e-30,gtol=1e-30,maxiter=10000,maxfev=10000),
                )    
            a = fit.x
        
        pack = fit.x.copy()
        for i,(a,b) in enumerate(bounds):
            if a is not None: pack[i] = max(a,pack[i])
            if b is not None: pack[i] = min(b,pack[i])
            
        transform, pack = result._unpack_transform(m, pack)
        distribution, pack = result._unpack_distribution(m, pack)
        y = result._apply_transform(x, transform)
        
        return result._with(
            optimization_result = fit,
            transform = transform,
            distribution = distribution,
            y = y
            )
        
        
        
        
        