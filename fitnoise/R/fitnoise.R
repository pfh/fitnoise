
ensure.columns.named <- function(mat, prefix) {
    if (is.null(colnames(mat))) {
        col.names <- character()
        for(i in seq_len(ncol(mat)))
            col.names[i] <- sprintf('%s%d', prefix, i)
        colnames(mat) <- col.names
    }
    mat
}


fitnoise.fit <- function(
        y, design, 
        model="Model_t_standard()",
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

    pyload()
    
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

