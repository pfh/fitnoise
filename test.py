
from rpy2 import robjects
from rpy2.robjects import r
from rpy2.robjects.numpy2ri import numpy2ri

import fitnoise

import numpy

n = 1000
m = 10
df = 5.0
dist = fitnoise.Mvt(
    numpy.zeros(m),
    numpy.identity(m),
    df
    )

design = numpy.ones((m,1))

data = numpy.array([ dist.random() for i in xrange(n) ])

factor = numpy.random.normal(size=m)
fac = numpy.outer(numpy.random.normal(size=n), factor)
data = data + fac

fit = fitnoise.Model_t_factors_standard(1)
#fit = fitnoise.Model_t_standard()
#fit = fitnoise.Model_t_factors_independent(1)
#fit = fitnoise.Model_t_independent()


fit = fit.fit_noise(data, design, use_theano=1, verbose=True)

fit = fit.fit_coef()

print fit

import pylab
pylab.plot(factor, fit.param.factors[0],'.')
pylab.show()

#print fitnoise.Model_t_standard().fit_noise(data, design)
#
#r.library("fitnoise")
#
#
#elist = r.new("EList",robjects.ListVector([('E',numpy2ri(data))]))
#
#rfit = r['fit.elist'](elist, numpy2ri(design))
#r['print'](rfit)
#
#print
#
#rpyfit = r['fit.elist'](elist, numpy2ri(design), python=True)
#r['print'](rpyfit)
