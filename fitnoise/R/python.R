
pyexec <- rPython::python.exec

pyload <- function() {
    pyexec("import fitnoise, numpy")
}

pyset <- function(name,value) {
    if (!is.null(dim(value))) {
        rPython::python.assign("_mat_vec",c(t(value)))
        rPython::python.assign("_mat_dim",dim(value))
        rPython::python.exec(sprintf(
            "%s = numpy.array(_mat_vec).reshape(_mat_dim)", 
            name))
    } else {
        rPython::python.assign(name, value)
    }
}

pyget <- function(name) {
    rPython::python.exec(sprintf("_mat = numpy.array(%s)",name))
    rPython::python.exec("_mat_vec = list(_mat.T.flat)")
    rPython::python.exec("_mat_dim = _mat.shape")
    vec <- rPython::python.get("_mat_vec")
    dim(vec) <- rPython::python.get("_mat_dim")
    vec
}