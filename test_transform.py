

import fitnoise
import numpy
import pylab

n = 1000
m = 2
df = 2.0
dist = fitnoise.Mvt(
    numpy.zeros(m),
    numpy.identity(m),
    df
    )

data = numpy.array([ dist.random() for i in xrange(n) ])

data = data*0.1 + numpy.random.random(size=n)[:,None] * 20.0

data = data + [-5.0,5.0]

x = 2.0 ** data
x = numpy.random.poisson(x)
#print x

for r in [ False, True ]:
 for t in [ 
      #'quadratic2', 
      'quadratic3' 
      ]:
  for d in [ 
      #'independent', 
      'uniformly_covariant', 
      #'allpairs_covariant' 
      ]:
    fit = fitnoise.transform(x, 
             transform=t, distribution=d, robust=r, verbose=True)
    print fit.optimization_result
    
    #print 'df',fit.distribution.df
    print
    print fit.transform
    print fit.distribution
    #print fit.x_median
    
    #print fit.transform[1] / numpy.sqrt(numpy.diag(fit.distribution.covar))
    
    y = fit.y
    pylab.plot(0.5*(y[:,0]+y[:,1]), y[:,1]-y[:,0],'.')

pylab.show()



