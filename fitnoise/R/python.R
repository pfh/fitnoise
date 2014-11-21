

# Oh. God.


pyexec <- rPython::python.exec

pyload <- function() {
    if (!exists(".fitnoise.python.loaded", envir=.GlobalEnv)) {
        pyexec("import numpy, json, urllib")
        pyexec("from fitnoise import *")
        assign(".fitnoise.python.loaded", TRUE, envir=.GlobalEnv)
    }
}


pyset <- function(name,value) {
    pyload()
    
    text <- RJSONIO::toJSON(value, digits=50)
    pyexec(
        sprintf(
            '%s = json.loads(urllib.unquote("%s"))',
            name, 
            URLencode(text)
            )
        )
    
    #if (!is.null(dim(value))) {
    #    pyset("_dim", 
    #    rPython::python.assign("_mat_vec",c(t(value)))
    #    rPython::python.assign("_mat_dim",dim(value))
    #    rPython::python.exec(sprintf(
    #        "%s = numpy.array(_mat_vec).reshape(_mat_dim)", 
    #        name))
    #} 
    #else {
    #    rPython::python.exec(sprintf("%s = json.loads(_json)",name))
    #}
}

pyset_single <- function(name,value) {
    pyset(name,value)
    pyexec(sprintf("%s = %s[0]",name,name))
}


pyget <- function(name) {
    pyload()
    
    #rPython::python.exec(sprintf("_mat = numpy.array(%s)",name))
    #rPython::python.exec("_mat_vec = list(_mat.T.flat)")
    #rPython::python.exec("_mat_dim = _mat.shape")
    #vec <- rPython::python.get("_mat_vec")
    #dim(vec) <- rPython::python.get("_mat_dim")
    
    pyexec(sprintf("_val = R_literal(%s)",name))
    text <- rPython::python.get("_val")
    pyexec("del _val")
    eval(parse(text=text))
}



