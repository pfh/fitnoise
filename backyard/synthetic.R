
library("limma")
library("fitnoise")

synth.elist <- function(
        group_size = 2,
        genes = 1000, 
        de_genes = 100, 
        de_amount = 10,
        df = 5.0
        ) {
    n <- genes
    m <- group_size * 2
    
    design <- cbind(
        rep(1, m),
        c(rep(0, group_size),rep(1, group_size))
        )
    
    random_normal <- matrix(
         rnorm(n*m),
         nrow=n
         )
    random_t <- random_normal * sqrt(df/rchisq(n, df=df))
    
    de <- sample(seq_len(n), de_genes)
    de_amount <- sample(c(-de_amount,de_amount), length(de), replace=T)
    E <- random_t
    for(i in seq_len(group_size)) {
        E[de, group_size+i] <- E[de, group_size+i] + de_amount
    }
    
    is_de <- rep(FALSE, n)
    is_de[de] <- TRUE
    
    elist <- new("EList",list(
            E = E
            ))
    
    list(
        elist = elist,
        design = design,
        test_coef = 2,
        is_de = is_de
        )
}


judge <- function(data) {
   if (!is.null(data$is_de)) {
       data$sensitivity <- sum(data$declared_de & data$is_de) /
           sum(data$is_de)
       data$fdr <- sum(data$declared_de & !data$is_de) / 
           max(1,sum(data$declared_de))
   }
   
   data
}


bootstrap <- function(data) {
    n <- nrow(data$elist)
    sampling <- sample(seq_len(n), replace=T)
    
    data$elist <- data$elist[sampling,]
    if (!is.null(data$is_de))
        data$is_de <- data$is_de[sampling]
    
    data
}


run.limma <- function(data, fdr) {
    fit <- lmFit(data$elist, data$design)
    fit <- eBayes(fit)
    tt <- topTable(
        fit, data$test_coef, 
        sort="none", number=nrow(fit)
        )

    data$fit <- fit
    data$toptable <- tt
    data$declared_de <- tt$adj.P.Val <= fdr
    data$declared_de[ is.na(data$declared_de) ] <- FALSE

    judge(data)
}


run.fitnoise <- function(data, fdr,  model="Model_t()") {
    data$fit <- fitnoise.fit(data$elist, data$design, model=model, verbose=T)
    data$fit <- fitnoise.test(data$fit, data$test_coef)
    
    data$declared_de <- data$fit$q.values <= fdr

    judge(data)
}



