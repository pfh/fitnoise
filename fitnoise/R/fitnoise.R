
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
        data, design,
        model="Model_t()",
        noise.design=NULL,
        control.design=NULL, controls=NULL,
        weights=NULL, counts=NULL,
        verbose=FALSE) {

    if (class(data) == "EList") {
        if (!is.null(data$weights)) {
            stopifnot(is.null(weights))
            weights <- data$weights
        }
        if (!is.null(data$other$counts)) {
            stopifnot(is.null(counts))
            counts <- data$other$counts
        }
        data <- data$E
    }

    if (is.null(noise.design))
        noise.design <- design

    pyexec("from fitnoise import *")


    data[is.na(data)] <- NaN

    pyset("data", data)
    pyexec("context = {}")
    if (!is.null(weights))
        pyset("context['weights']", weights)
    if (!is.null(counts))
        pyset("context['counts']", counts)
    pyexec("dataset = fitnoise.Dataset(data, context)")

    pyset("design", design)
    pyset("noise_design", noise.design)
    pyset("control_design", control.design)
    pyset("controls", controls)
    pyset_scalar("verbose", verbose)

    pyexec(sprintf("fit = %s", model))
    pyexec("fit = fit.fit(
        dataset,
        design=design,
        noise_design=noise_design,
        control_design=control_design,
        controls=controls,
        verbose=verbose,
        )")

    list(
        pyfit = pyref("fit"),
        description = pyget("fit.description"),
        param = pyget("as_jsonic( fit.param )"),
        score = pyget("as_jsonic([ fit.score ])"),
        coef = pyget("as_jsonic( fit.coef )"),
        noise.p.values = pyget("as_jsonic( fit.noise_p_values )"),
        noise.combined.p.value = pyget("as_jsonic([ fit.noise_combined_p_value ])"),
        averages = pyget("as_jsonic( fit.averages )")
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
    fit$description <- pyget("fit.description")

    fit$contrasts <- pyget("as_jsonic( fit.contrasts )")
    if (!is.null(colnames(contrasts)))
        colnames(fit$contrasts) <- colnames(contrasts)

    fit$p.values <- pyget("as_jsonic( fit.p_values )")
    fit$q.values <- pyget("as_jsonic( fit.q_values )")

    fit
}



fitnoise.weights <- function(fit) {
    pyset("fit", fit$pyfit)
    pyget("as_jsonic( fit.get_weights() )")
}



fitnoise.transform <- function(
        x, design=NULL, order=2,
        verbose=FALSE) {

    pyexec("from fitnoise import *")

    pyset("x",x)
    pyset("design",design)
    pyset_scalar("order",order)
    pyset_scalar("verbose",verbose)

    pyexec("fit = transform(
        x=x,
        design=design,
        order=order,
        verbose=verbose,
        )")

    y <- pyget("as_jsoinc( fit.y )")
    y <- mimic.matrix.names(y, x)

    description <- pyget("repr(fit)")

    list(
        x = x,
        design = design,
        description = description,
        y = y
        )
}
