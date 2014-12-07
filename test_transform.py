

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

data = data*2.0 + numpy.random.random(size=n)[:,None] * 20.0

data = data -[5.0,10.0] #+ ([-5.0,5.0]+[0.0]*(m-2))

x = 2.0 ** data
x = numpy.random.poisson(x)

pylab.subplot(211)
for o in [ 2 ]:
    fit = fitnoise.transform(x, 
             order=o, verbose=True)
    
    y = fit.y
    pylab.plot(0.5*(y[:,0]+y[:,1]), y[:,1]-y[:,0],'.')

pylab.subplot(212)
xx = numpy.arange(20)
xxt = numpy.tile(xx[:,None], (1,m))
#pylab.plot(xx, numpy.sqrt(xx))
#pylab.plot(xx, numpy.log(xx)/numpy.log(2.0))

yy = fit._apply_transform(fit.transform,xxt)
for i in xrange(m):
    pylab.plot(xx,yy[:,i])

pylab.show()



