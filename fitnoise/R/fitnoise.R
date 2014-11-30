
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
        model="fitnoise.Model_t_standard()",
        noise.design=NULL, 
        control.design=NULL, controls=NULL, 
        weights=NULL, counts=NULL) {
            
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
    
    pyset("y", y)
    pyexec("context = {}")
    if (!is.null(weights))
        pyset("context['weights']", weights)
    if (!is.null(counts))
        pyset("context['counts']", counts)
    pyexec("dataset = Dataset(y, context)")
    
    pyset("design", design)
    pyset("noise_design", noise.design)
    pyset("control_design", control.design)
    pyset("controls", controls)
    
    pyexec(sprintf("fit = %s", model))
    pyexec("fit = fit.fit(
        dataset,
        design=design,
        noise_design=noise_design,
        control_design=control_design,
        controls=controls,
        )")
    
    result = list(
        )
}


fitnoise.transform <- function(
        x, design=NULL, transform="varstab2",
        verbose=FALSE) {
    
    pyset("x",x)
    pyset("design",design)
    pyset_scalar("transform",transform)
    pyset_scalar("verbose",verbose)
    
    pyexec("fit = fitnoise.transform(
        x=x,
        design=design,
        transform=transform,
        verbose=verbose,
        )")

    y <- pyget("fit.y.tolist()")
    y <- mimic.matrix.names(y, x)
    
    fit <- pyget("repr(fit)")

    pyexec("del x, design, transform, verbose, fit")

    list(
        x = x,
        design = design,
        fit = fit,
        y = y
        )
}




