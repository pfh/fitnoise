
import numpy

import fitnoise

p = numpy.random.random(10)

print p

print fitnoise.fdr(p)

from rpy2 import robjects
from rpy2.robjects import r
from rpy2.robjects.numpy2ri import numpy2ri

r["library"]("limma")
print numpy.array( r["p.adjust"](numpy2ri(p), method="BH") )