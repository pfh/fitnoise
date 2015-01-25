
ensure.matrix.columns.named <- function(mat, prefix) {
    if (is.null(colnames(mat))) {
        col.names <- character()
        for(i in seq_len(ncol(mat)))
            col.names[i] <- sprintf('%s%d', prefix, i)
        colnames(mat) <- col.names
    }
    mat
}

mimic.matrix.names <- function(x, template) {
    x <- as.matrix(x)
    rownames(x) <- rownames(template)
    colnames(x) <- colnames(template)
    x
}


fitnoise.fit <- function(
        y, design, 
        model="Model_t()",
        noise.design=NULL, 
        control.design=NULL, controls=NULL, 
        weights=NULL, counts=NULL,
        verbose=FALSE) {
            
    if (class(y) == "EList") {
        if (!is.null(y$weights)) {
            stopifnot(is.null(weights))
            weights <- y$weights
        }
        if (!is.null(y$other$counts)) {
            stopifnot(is.null(counts))
            counts <- y$other$counts
        }
        y <- y$E
    }
    
    if (is.null(noise.design))
        noise.design <- design
    
    pyexec("from fitnoise import *")
    
    pyset("y", y)
    pyexec("context = {}")
    if (!is.null(weights))
        pyset("context['weights']", weights)
    if (!is.null(counts))
        pyset("context['counts']", counts)
    pyexec("dataset = fitnoise.Dataset(y, context)")
    
    pyset("design", design)
    pyset("noise_design", noise.design)
    pyset("control_design", control.design)
    pyset("controls", controls)
    pyset_scalar("verbose", verbose)
        
    pyexec(sprintf("fit = %s", model))
    pyexec("fit = (
        fit
        .fit_noise(
            dataset,
            design=noise_design,
            control_design=control_design,
            controls=controls,
            verbose=verbose,
            )
        .fit_coef(design)
        )")
    
    list(
        pyfit = pyref("fit"),
        description = pyget("repr(fit)"),
        param = pyget("fit.param.as_jsonic()"),
        score = pyget("fit.score")
        )
}



fitnoise.test <- function(
        fit, coef=NULL, contrasts=NULL
        ) {
    
    if (!is.null(coef))
        coef <- coef - 1  #Python is zero based
    
    if (!is.null(contrasts))
        contrasts <- as.matrix(contrasts)
    
    pyset("fit", fit$pyfit)
    pyset("coef", coef)
    pyset("contrasts", contrasts)
    
    pyexec("fit = fit.test(coef=coef,contrasts=contrasts)")
    
    fit$pyfit <- pyref("fit")
    fit$description <- pyget("repr(fit)")  
    
    fit$contrasts <- pyget("fit.contrasts.tolist()")
    if (!is.null(colnames(contrasts)))
        colnames(fit$contrasts) <- colnames(contrasts)
    
    fit$p.values <- pyget("fit.p_values.tolist()")
    fit$q.values <- pyget("fit.q_values.tolist()")
    
    fit
}


fitnoise.transform <- function(
        x, design=NULL, order=2,
        verbose=FALSE) {
    
    pyset("x",x)
    pyset("design",design)
    pyset_scalar("order",order)
    pyset_scalar("verbose",verbose)
    
    pyexec("fit = fitnoise.transform(
        x=x,
        design=design,
        order=order,
        verbose=verbose,
        )")

    y <- pyget("fit.y.tolist()")
    y <- mimic.matrix.names(y, x)
    
    fit <- pyget("repr(fit)")

    list(
        x = x,
        design = design,
        fit = fit,
        y = y
        )
}




