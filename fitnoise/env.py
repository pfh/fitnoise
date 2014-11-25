
import warnings, copy

have_theano = False
try:
    import theano
    from theano import tensor
    from theano import gradient
    try:
        from theano.tensor import nlinalg, slinalg
        have_theano = True
    except:
        warnings.warn("Couln't import linear algebra for theano, need a more recent version")
except:
    pass


import numpy, numpy.linalg, numpy.random
import scipy, scipy.optimize, scipy.stats, scipy.special



class Withable(object):
    def _with(self, **kwargs):
        result = copy.copy(self)
        for name in kwargs:
            setattr(result,name,kwargs[name])
        return result


def is_theanic(x):
    return have_theano and isinstance(x, tensor._tensor_py_operators)

def as_tensor(x, dtype='float64'):
    if not isinstance(x, numpy.ndarray) and \
       not is_theanic(x):
        x = numpy.array(x)            

    if not x.dtype == dtype:
        x = x.astype(dtype)

    return x

    
def as_scalar(x, dtype='float64'):
    x = as_tensor(x, dtype)
    assert x.ndim == 0
    return x


def as_vector(x, dtype='float64'):
    x = as_tensor(x, dtype)
    assert x.ndim == 1
    return x


def as_matrix(x, dtype='float64'):
    x = as_tensor(x, dtype)
    assert x.ndim == 2
    return x


def dot(a,b):
    a = as_tensor(a)
    b = as_tensor(b)
    return (
        tensor.dot if is_theanic(a) or is_theanic(b) 
        else numpy.dot
        )(a,b)


def tensordot(a,b,axes):
    a = as_tensor(a)
    b = as_tensor(b)
    return (
        tensor.tensordot if is_theanic(a) or is_theanic(b) 
        else numpy.tensordot
        )(a,b,axes)


def outer(a,b):
    a = as_vector(a)
    b = as_vector(b)
    return (
        tensor.outer if is_theanic(a) or is_theanic(b) 
        else numpy.outer
        )(a,b)


def take(a,indices,axis):
    if is_theanic(a) or is_theanic(indices):
        return tensor.take(a,indices,axis)
    else:
        return numpy.take(a,indices,axis)


def take2(a,i0,i1):
    return take(take(a,i0,0),i1,1)


def diag(x):
    x = as_tensor(x)
    return tensor.diag(x) if is_theanic(x) else numpy.diag(x)


def zeros(shape):
    any_t = any(is_theanic(item) for item in shape)
    return tensor.zeros(shape) if any_t else numpy.zeros(shape)

def ones(shape):
    any_t = any(is_theanic(item) for item in shape)
    return tensor.ones(shape) if any_t else numpy.ones(shape)


def concatenate(items):
    any_t = any(is_theanic(item) for item in items)
    return tensor.concatenate(items) if any_t else numpy.concatenate(items)


def log(x):
    x = as_tensor(x)
    return tensor.log(x) if is_theanic(x) else numpy.log(x)


def exp(x):
    x = as_tensor(x)
    return tensor.exp(x) if is_theanic(x) else numpy.exp(x)


#From Numerical Recipes in C
def lanczos_gammaln(x):
    x = as_tensor(x)
    y = x
    tmp = x + 5.5
    tmp = tmp - (x+0.5)*log(tmp)
    ser = 1.000000000190015
    for cof in (
        76.18009172947146, 
        -86.50532032941677, 
        24.01409824083091, 
        -1.231739572450155, 
        0.1208650973866179e-2, 
        -0.5395239384953e-5,
        ):
        y = y + 1
        ser = ser + cof / y
    return -tmp + log(2.5066282746310005*ser/x)


def gammaln(x):
    x = as_tensor(x)
    return lanczos_gammaln(x) if is_theanic(x) else scipy.special.gammaln(x)


def inverse(A):
    A = as_matrix(A)
    return (nlinalg.matrix_inverse if is_theanic(A) else numpy.linalg.inv)(A)


def det(A):
    A = as_matrix(A)
    return (nlinalg.det if is_theanic(A) else numpy.linalg.det)(A)


#Doesn't have gradient
def cholesky(A):
    A = as_matrix(A)
    return (slinalg.cholesky if is_theanic(A) else numpy.linalg.cholesky)(A)


#Doesn't have gradient
def qr_complete(A):
    A = as_matrix(A)
    if is_theanic(A):
        return nlinalg.QRFull('complete')(A)
    else:
        return numpy.linalg.qr(A, mode="complete")




