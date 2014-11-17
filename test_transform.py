

import fitnoise
from fitnoise import transform

import numpy

n = 1000
m = 2
df = 5.0
dist = fitnoise.Mvt(
    numpy.zeros(m),
    numpy.identity(m),
    df
    )

data = numpy.array([ dist.random() for i in xrange(n) ])

data = data*0.05 + numpy.random.random(size=n)[:,None] * 20.0

data = data #+ [4.0,5.0]

x = 2.0 ** data
x = numpy.random.poisson(x)# + 2.0*numpy.random.random(size=x.shape))
#print x


fit = transform.Transform_to_t().fit(x, verbose=True)
print fit.optimization_result

#print 'df',fit.distribution.df
print
print fit.transform
print 1/fit.transform[1]
#print fit.transform.a
#print fit.transform.b
#print fit.transform.c
#print fit.distribution.mean
print fit.distribution.covar
#print fit.x_median

import pylab
#pylab.plot(jity[:,0]+jity[:,1], jity[:,1]-jity[:,0],'.')
#pylab.plot(numpy.argsort(numpy.argsort(fit.y[:,0]+fit.y[:,1])), fit.y[:,1]-fit.y[:,0],'.',color="red")

lx = numpy.log(x) / numpy.log(2.0)
#pylab.plot(0.5*(lx[:,0]+lx[:,1]), lx[:,1]-lx[:,0],'.',color="blue")

pylab.plot(0.5*(fit.y[:,0]+fit.y[:,1]), fit.y[:,1]-fit.y[:,0],'.',color="red")
#pylab.plot(fit.y[:,0],fit.y[:,1],'.')
pylab.show()
