
from .env import *

def _mahalanobis(covar, offset):
    return (offset * 
            tensordot(offset,inverse(covar),((offset.ndim-1,),(1,)))
            ).sum(axis=offset.ndim-1)

def _indent(text, n):
    return text.replace("\n", "\n"+" "*n)
            
# Agnostic as to whether numpy or theano
# Mvnormal(dvector('mean'), dmatrix('covar'))
#
# TODO: maybe store covar as cholesky decomposition
#
class Mvnormal(object):
    def __init__(self, mean, covar):
        self.mean = as_vector(mean)
        self.covar = as_matrix(covar)


    def __repr__(self):
        return "Mvnormal(\n   mean=%s,\n  covar=%s\n)" % (
            _indent(repr(self.mean),8),
            _indent(repr(self.covar),8)
            ) 


    # Not available for theano
    @property
    def good(self):
        result = numpy.isfinite(self.mean)
        for i in xrange(len(self.covar)):
            result = result & numpy.isfinite(self.covar[i])
        return result
    
    
    def log_density(self, x):
        x = as_tensor(x)
        offset = x - self.mean
        n = self.mean.shape[0]
        
        #TODO: cleverer linear algebra
        return -0.5*(
            log(2*numpy.pi)*n
            + log(det(self.covar))
            + _mahalanobis(self.covar, offset)
            )

    
    def p_value(self, x):
        x = as_vector(x)
        offset = x - self.mean
        df = self.covar.shape[0]
        q = dot(offset, dot(inverse(self.covar), offset))
        return scipy.stats.chi2.sf(q, df=df)
    
    
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

            
    def plus_covar(self, A):
        A = as_matrix(A)
        return Mvnormal(self.mean, self.covar+A)


    def shifted(self, x):
        x = as_vector(x)
        return Mvnormal(self.mean+x, self.covar)


    def marginal(self, i):
        i = as_vector(i, 'int32')
        return Mvnormal(take(self.mean,i,0), take2(self.covar,i,i))


    def conditional(self, i1,i2,x2):
        i1 = as_vector(i1, 'int32')
        i2 = as_vector(i2, 'int32')
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


    def __repr__(self):
        return "Mvt(\n     df=%s,\n   mean=%s,\n  covar=%s\n)" % (
            repr(self.df),
            _indent(repr(self.mean),8),
            _indent(repr(self.covar),8)
            ) 


    @property
    def good(self):
        result = numpy.isfinite(self.mean)
        for i in xrange(len(self.covar)):
            result = result & numpy.isfinite(self.covar[i])
        return result


    def log_density(self, x):
        x = as_tensor(x)
        offset = x - self.mean
        p = self.covar.shape[0]
        v = self.df
        distance2 = _mahalanobis(self.covar, offset)
        return (
            gammaln(0.5*(v+p))
            - gammaln(0.5*v)
            - (0.5*p)*log(numpy.pi*v)
            - 0.5*log(det(self.covar))
            - (0.5*(v+p))*log(1+distance2/v)
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
        x = as_vector(x)
        return Mvt(self.mean+x, self.covar, self.df)

    
    def plus_covar(self, A):
        A = as_matrix(A)
        return Mvt(self.mean, self.covar+A, self.df)


    def marginal(self, i):
        i = as_vector(i, 'int32')
        return Mvt(take(self.mean,i,0), take2(self.covar,i,i), self.df)


    def conditional(self, i1,i2,x2):
        i1 = as_vector(i1, 'int32')
        i2 = as_vector(i2, 'int32')
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
