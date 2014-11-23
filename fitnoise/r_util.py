
import numpy

def R_literal(item):
    if item is None:
        return 'NULL'
    elif isinstance(item, str):
        return "'" + item.replace('\\','\\\\').replace("'","\\'") + "'"
    elif isinstance(item, unicode):
        return R_literal(item.encode("utf-8"))
    elif isinstance(item, bool):
        return 'TRUE' if item else 'FALSE'
    elif isinstance(item, float):
        return repr(item)
    elif isinstance(item, int):
        return '%d' % item
    elif isinstance(item, list) or isinstance(item, tuple):
        return 'list(' + ',\n'.join( R_literal(subitem) for subitem in item ) + ')'
    elif isinstance(item, numpy.ndarray) and item.ndim == 1:
        return 'c(' + ',\n'.join( R_literal(subitem) for subitem in item ) + ')'
    elif isinstance(item, numpy.ndarray) and item.ndim == 2:
        return 'matrix(%s,nrow=%d,ncol=%d,byrow=TRUE)' % (
            R_literal(item.flatten()),
            item.shape[0],
            item.shape[1],
            )
    else:
        assert False, "Can't encode %s" % repr(item)
