
from rpy2 import robjects
from rpy2.robjects import r
from rpy2.robjects.numpy2ri import numpy2ri

import fitnoise

import numpy

r.library("fitnoise")

n = 10
m = 4
df = 5.0
dist = fitnoise.Mvt(
    numpy.zeros(m),
    numpy.identity(m),
    df
    )

design = numpy.ones((m,1))

data = numpy.array([ dist.random() for i in xrange(n) ])

print fitnoise.model_t_standard.fit_noise(data, design)

elist = r.new("EList",robjects.ListVector([('E',numpy2ri(data))]))

rfit = r['fit.elist'](elist, numpy2ri(design))
r['print'](rfit)

rpyfit = r['fit.elist'](elist, numpy2ri(design), python=True)
r['print'](rpyfit)
