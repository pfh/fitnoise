

# Oh. God.

pyquote <- function(text) {
    text <- gsub("\\\\","\\\\\\\\",text)
    text <- gsub("\n","\\\\n",text)
    text <- gsub("\"","\\\\\"",text)
    sprintf('"%s"',text)
}


pyexec <- function(text) {
    pyload()
    rPython::python.exec(text)
}


pyload <- function() {
    if (!exists(".fitnoise.python.loaded", envir=.GlobalEnv)) {
        assign(".fitnoise.python.loaded", TRUE, envir=.GlobalEnv)
        pyexec("import numpy, json, urllib, fitnoise")
    }
}


pyset_none <- function(name) {
    pyexec(sprintf("%s = None", name))
}


pyset <- function(name,value) {
    if (is.null(value)) {
        pyset_none(name)
        return()
    }
    
    text <- jsonlite::toJSON(value, digits=50)
    pyexec(
        sprintf(
            '%s = numpy.array(json.loads(%s))',
            name, 
            pyquote(text)
            )
        )
}


pyset_scalar <- function(name,value) {
    pyset(name,value)
    if (is.null(value)) return()

    pyexec(sprintf("%s = %s[0]",name,name))
}


pyget <- function(name) {
#    pyexec(sprintf("_val = fitnoise.R_literal(%s)",name))
    pyexec(sprintf("_val = json.dumps(%s)",name))
    text <- rPython::python.get("_val")
    pyexec("del _val")
#    eval(parse(text=text))
    jsonlite::fromJSON(text)
}



