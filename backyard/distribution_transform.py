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

def pull_out_strict_triangle(vec, n):
    pieces = [ ]
    for i in xrange(0,n):
        pieces.append(vec[:i])
        pieces.append(zeros((n-i,)))
        vec = vec[i:]
    return concatenate(pieces).reshape((n,n)), vec

#print pull_out_triangle(numpy.array([1,2,3,4,5,6]), 3)


class Transform(Withable):
    def _configured(self):
        return self
    
    def fit(self, x, design=None, verbose=False):
        x = as_matrix(x)
        n,m = x.shape
        
        if design is None:
           design = ones((m,1))
        design = as_matrix(design)
        
        
        q,r = qr_complete(design)
        null = q[:,design.shape[1]:]
        
    
        result = self._with(
            x=x,
            design=design,
            null=null,
            )._configured()
    
        initial = numpy.concatenate([result._param_initial,result._dist_initial])    
        bounds = result._param_bounds + result._dist_bounds
        
        lower = numpy.array([ b[0] if b[0] is not None else -1e30 for b in bounds])
        upper = numpy.array([ b[1] if b[1] is not None else 1e30 for b in bounds])
        
        vpack_raw = tensor.dvector("pack_raw")
        vlower = tensor.dvector("lower")
        vupper = tensor.dvector("upper")
        
        vpack = vpack_raw.clip(vlower,vupper)
        
        #vpack = tensor.dvector("pack")
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
        eps = 1e-4
        vy = result._apply_transform(vparam, vx)
        vderiv = (result._apply_transform(vparam, vx+eps) - vy)/eps
        
        #vtransform_cost = -log(vderiv).sum()  
        vtransform_cost = -0.5 * log(dot(null.T,vderiv.T)**2).sum()  
        
        #vdist_cost = -vdist.log_density(vy).sum()
        vdist_cost = -(
            vdist
            .transformed(null.T)
            .log_density(dot(null.T,vy.T).T)
            .sum()
            )
        
        vcost = vtransform_cost + vdist_cost + vx.shape[0]*((vpack-vpack_raw)**2).sum() 
        vcost_grad = gradient.grad(vcost, vpack)
        
        func = theano.function(
            [vpack_raw, vlower, vupper, vx],
            [vcost] #, vcost_grad],
            #on_unused_input="ignore",
            )    
        
        def score(pack):
            #old = pack
            #pack = pack.copy()
            #for i,(a,b) in enumerate(bounds):
            #    if a is not None: pack[i] = max(a,pack[i])
            #    if b is not None: pack[i] = min(b,pack[i])
            #err = numpy.absolute(old-pack).sum()
            result = func(pack,lower,upper,x)
            #if verbose:
            #    print result[0]
            return result[0] #+ (1.0+abs(result[0]))*err
        
        a = initial
        last = None
        for i in xrange(100):
            fit = scipy.optimize.minimize(
                score, 
                a,
                method="Nelder-Mead", 
                #method="L-BFGS-B",
                #jac=True,
                #bounds=bounds,
                options=dict(maxfev=10000),
                )    
            a = fit.x
            if verbose: print fit.fun
            if last is not None and last-fit.fun <= 1e-10: break
            last = fit.fun
        
        pack = fit.x.copy()
        for i,(a,b) in enumerate(bounds):
            if a is not None: pack[i] = max(a,pack[i])
            if b is not None: pack[i] = min(b,pack[i])

        transform, pack = result._unpack_transform(m, pack)
        distribution, pack = result._unpack_distribution(m, pack)
        y = result._apply_transform(transform, x)
        y = y - distribution.mean
        
        adjustment = log(1e6 / x.sum(axis=0)).mean()/log(2.0)
        y_per_million = y + adjustment
        
        return result._with(
            optimization_result = fit,
            transform = transform,
            distribution = distribution,
            y = y,
            y_per_million = y_per_million,
            )


class Transform_mixin_varstab3(object):
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
        result = super(Transform_mixin_varstab3, self)._configured()
        m = result.x.shape[1]
        
        x_median = numpy.median(result.x,axis=0)
        guess_b = x_median+1
        print 'guess_b', guess_b
        
        param_initial = (
             list(guess_b) +
             [0.0]*m 
             )
        
        param_bounds = (
            [(1e-10, None)]*m+
            [(0.0, None)]*m
            )
        assert len(param_initial) == len(param_bounds)
        return result._with(
            _param_initial = param_initial,
            _param_bounds = param_bounds,
            )


class Transform_mixin_quadratic(object):
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
        result = super(Transform_mixin_quadratic, self)._configured()
        m = result.x.shape[1]
        
        x_median = numpy.median(result.x,axis=0)
        guess_b = x_median+1
        print 'guess_b', guess_b
        
        param_initial = (
             list(guess_b) +
             list(guess_b**2) 
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


class Transform_mixin_cubic(object):
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
        result = super(Transform_mixin_cubic, self)._configured()
        m = result.x.shape[1]
        
        x_median = numpy.median(result.x,axis=0)
        guess_b = x_median+1
        print 'guess_b', guess_b
        
        param_initial = (
             list(guess_b) +
             list(guess_b**2) +
             list(guess_b**3)
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



class Transform_mixin_varstab2(object):
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
        result = super(Transform_mixin_varstab2, self)._configured()
        m = result.x.shape[1]
        
        x_median = numpy.median(result.x,axis=0)
        
        guess_b = x_median+1
        print 'guess_b', guess_b
        
        param_initial = (
             list(guess_b)
             )
        
        param_bounds = (
            [(1e-10, None)]*m
            )
        assert len(param_initial) == len(param_bounds)
        return result._with(
            _param_initial = param_initial,
            _param_bounds = param_bounds,
            )


class Transform_mixin_independent(object):        
    def _unpack_distribution(self, m, pack):
        m = self.x.shape[1]
        mean, pack = pack[:m], pack[m:]
        v, pack = pack[0],pack[1:]
        covar = v*numpy.identity(m)
        return distributions.Mvnormal(mean, covar), pack
    
    def _configured(self):
        result = super(Transform_mixin_independent,self)._configured()
        
        m = result.x.shape[1]
        
        y = result._apply_transform(
            result._unpack_transform(m,numpy.array(result._param_initial))[0], 
            result.x)
        guess_mean = numpy.mean(y, axis=0)
        guess_var = numpy.var(y.flat)
        
        print "guess_mean", guess_mean
        print "guess_var", guess_var
        
        dist_initial = list(guess_mean) + [guess_var]
        dist_bounds = [(None,None)]*m + [(1e-6,None)]
        
        return result._with(
            _dist_initial = dist_initial,
            _dist_bounds = dist_bounds,
            )


class Transform_mixin_uniformly_covariant(object):        
    def _unpack_distribution(self, m, pack):
        #TODO: allow m to be theanic
        m = self.x.shape[1]
        mean, pack = pack[:m], pack[m:]
        v, pack = pack[0],pack[1:]
        cv, pack = pack[0],pack[1:]
        covar = v*(numpy.identity(m)*(1.0-cv) + cv)
        #covar = numpy.identity(m) + L + L.T
        return distributions.Mvnormal(mean, covar), pack
    
    def _configured(self):
        result = super(Transform_mixin_uniformly_covariant,self)._configured()
        
        m = result.x.shape[1]
        
        y = result._apply_transform(
            result._unpack_transform(m,numpy.array(result._param_initial))[0], 
            result.x)
        guess_mean = numpy.mean(y, axis=0)
        guess_var = numpy.var(y.flat)
        
        print "guess_mean", guess_mean
        print "guess_var", guess_var
        
        dist_initial = list(guess_mean) + [guess_var, 0.75]
        dist_bounds = [(None,None)]*m + [(1e-6,None), (0.0,0.9999)]
        
        return result._with(
            _dist_initial = dist_initial,
            _dist_bounds = dist_bounds,
            )

        
class Transform_mixin_allpairs_covariant(object):        
    def _unpack_distribution(self, m, pack):
        #TODO: allow m to be theanic
        m = self.x.shape[1]
        mean, pack = pack[:m], pack[m:]
        v, pack = pack[0],pack[1:]
        L, pack = pull_out_strict_triangle(pack, m)
        covar = v*(numpy.identity(m) + L + L.T)
        #covar = numpy.identity(m) + L + L.T
        return distributions.Mvnormal(mean, covar), pack
    
    def _configured(self):
        result = super(Transform_mixin_allpairs_covariant,self)._configured()
        
        m = result.x.shape[1]
        
        y = result._apply_transform(
            result._unpack_transform(m,numpy.array(result._param_initial))[0], 
            result.x)
        guess_mean = numpy.mean(y, axis=0)
        guess_var = numpy.var(y.flat)
        
        print "guess_mean", guess_mean
        print "guess_var", guess_var
        
        dist_initial = list(guess_mean) + [guess_var]
        dist_bounds = [(None,None)]*m + [(1e-6,None)]
        for i in xrange(m):
            dist_initial.extend([0.75]*i)
            dist_bounds.extend([(0.0,0.9999)]*i)
        
        return result._with(
            _dist_initial = dist_initial,
            _dist_bounds = dist_bounds,
            )


class Transform_mixin_t(object):
    def _unpack_distribution(self, m, pack):
        df, pack = pack[0], pack[1:]
        dist, pack = super(Transform_mixin_t,self)._unpack_distribution(m, pack)
        return distributions.Mvt(dist.mean, dist.covar, df), pack

    def _configured(self):
        result = super(Transform_mixin_t,self)._configured()
        return result._with(
            _dist_initial = [ 5.0 ] + result._dist_initial,
            _dist_bounds = [(1.0,1000.0)] + result._dist_bounds,
            )




transform_mixins = { 
    "quadratic" : Transform_mixin_quadratic,
    "cubic" : Transform_mixin_cubic,
    "varstab3" : Transform_mixin_varstab3,
    "varstab2" : Transform_mixin_varstab2,
    }

distribution_mixins = {
    "allpairs_covariant" : Transform_mixin_allpairs_covariant,
    "uniformly_covariant" : Transform_mixin_uniformly_covariant,
    "independent" : Transform_mixin_independent,
    }


def transform(x, 
        design=None,
        transform="quadratic", 
        distribution="uniformly_covariant", 
        robust=True, 
        verbose=False):
    class T( 
        distribution_mixins[distribution], 
        transform_mixins[transform], 
        Transform): pass

    if robust:
        class T(Transform_mixin_t, T): pass
    
    return T().fit(x, design=design, verbose=verbose)



        