
import numpy

def fdr(p): 
    """ Benjamini & Hochberg FDR q-values """
    
    p = numpy.array(p, dtype='float64')
    p[numpy.isnan(p)] = 1.0
    
    order = numpy.argsort(p)[::-1]
    rorder = numpy.argsort(order)    
    psorted = p[order]

    qsorted = numpy.minimum(
        1.0,
        numpy.minimum.accumulate(
            psorted * len(psorted) / numpy.arange(len(psorted),0,-1)
            )
        )
        
    q = qsorted[rorder]
    return q

