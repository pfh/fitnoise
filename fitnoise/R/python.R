

# Oh. God.

.fitnoise.env <- new.env()
.fitnoise.env$loaded <- FALSE
.fitnoise.env$counter <- 0

pyquote <- function(text) {
    text <- gsub("\\\\","\\\\\\\\",text)
    text <- gsub("\n","\\\\n",text)
    text <- gsub("\"","\\\\\"",text)
    sprintf('"%s"',text)
}


pyexec <- function(text) {
    text = sprintf(
        "try: %s\nexcept:\n  traceback.print_exc()\n  raise",
        text)
      

    pyload()
    rPython::python.exec(text)
}


pyload <- function() {
    if (!.fitnoise.env$loaded) {
        .fitnoise.env$loaded <- TRUE
        pyexec("import numpy, json, urllib, traceback")
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
    
    if (is.environment(value)) {
        pyexec(sprintf("%s = %s", name, value$name))
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


pyref <- function(text) {
    env <- new.env()
    
    .fitnoise.env$counter <- .fitnoise.env$counter + 1
    env$name <- sprintf("_pyref_%d", .fitnoise.env$counter)
    
    pyexec(sprintf("%s = (%s)", env$name, text))
    
    reg.finalizer(env, function(env) {
       cat("Freeing",env$name,"\n")
       pyexec(sprintf("del %s", env$name)) 
    })
    
    env
}


