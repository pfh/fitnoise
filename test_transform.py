

import fitnoise
import numpy
import pylab

n = 10000
m = 2
df = 10.0
dist = fitnoise.Mvt(
    numpy.zeros(m),
    numpy.identity(m),
    df
    )

data = numpy.array([ dist.random() for i in xrange(n) ])

data = data*0.1 + numpy.random.random(size=n)[:,None] * 20.0

data = data -[10.0,10.0] #+ ([-5.0,5.0]+[0.0]*(m-2))

x = 2.0 ** data
x = numpy.random.poisson(x)
#print x

for t in [ 
      #'pow',
      #'arcsinh',
      'linear',
      'quadratic',
      "cubic" 
      #'varstab2', 
      #'varstab3',
      ]:
    fit = fitnoise.transform(x, 
             transform=t, verbose=True)
    #print fit.optimization_result
    
    #print 'df',fit.distribution.df
    print
    print fit.transform
    #print fit.x_median
    
    #print fit.transform[1] / numpy.sqrt(numpy.diag(fit.distribution.covar))
    
    y = fit.y
    pylab.plot(0.5*(y[:,0]+y[:,1]), y[:,1]-y[:,0],'.')

pylab.show()



