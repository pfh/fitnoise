
# Check if fitnoise with forced hyperparameter choices exactly matches limma.

#Unmoderated version of limma 
#(by changing the value returned by fitFDist)
#library(unmodlimma)

library(limma)

library(fitnoise)

perform.test <- function(anova, weighted) {
    set.seed(2015)
    
    n <- 100
    m <- 6
    c <- 3
    if (anova) {
        test.coef <- c(1,2)
        null.coef <- c(3)
    } else {
        test.coef <- c(1)
        null.coef <- c(2,3)
    }
    X <- matrix(rnorm(m*c), nrow=m)
    
    #Force X to be orthogonal
    X <- qr.Q(qr(X))
    
    y <- matrix(rnorm(n*m), nrow=n)
    y <- y / sqrt(rchisq(n, df=5.0))
    
    if (weighted)
        weights <- matrix(runif(n*m), nrow=n)
    else
        weights <- matrix(rep(1,n*m), nrow=n)
    
    
    lfit <- lmFit(y, X, weights=weights)
    lfit <- eBayes(lfit)
    
    #lcont <- eBayes(contrasts.fit(lfit, coefficients=test.coef))
    
    cat("df.prior =",lfit$df.prior,"\n")
    cat("s2.prior =",lfit$s2.prior,"\n")
    
    ffit <- fitnoise.fit(y, X, weights=weights, 
        force.param=c(lfit$df.prior, lfit$s2.prior))
    
    cat("Max discrepancy in coefficients     ",
        max(abs(ffit$coef - lfit$coefficients)),
        "\n")
    
    ltop <- topTable(lfit, coef=test.coef, sort.by="none", number=n)
    ftop <- fitnoise.test(ffit, coef=test.coef)
    
    cat("Max discrepancy in p values         ",
        max(abs(ftop$p.values - ltop$P.Value)),
        "\n")
    
    cat("Max discrepancy in adjusted p values",
        max(abs(ftop$q.values - ltop$adj.P.Val)),
        "\n")
    

    anova.p <- c()
    for(i in 1:n) {
       an <- anova(
           lm(y[i,] ~ X+0,weights=weights[i,]), 
           lm(y[i,] ~ X[,null.coef,drop=F]+0,weights=weights[i,])
           )
       anova.p[i] <- an$"Pr(>F)"[2]
    }
    
    cat("Max discrepancy Fitnoise to unmoderated ",
        max(abs(ftop$p.values - anova.p)),
        "\n")
    
    cat("Max discrepancy limma to unmoderated ",
        max(abs(ltop$P.Value - anova.p)),
        "\n")

    cat("\n")
    
    list(y=y,X=X,weights=weights,
         lfit=lfit,ltop=ltop, 
         ffit=ffit,ftop=ftop,
         anova.p=anova.p)
}

cat("Unweighted, test single coefficient\n")
result1 <- perform.test(anova=F, weighted=F)

cat("Weighted, test single coefficient\n")
result2 <- perform.test(anova=F, weighted=T)

cat("Unweighted, test multiple coefficients\n")
result3 <- perform.test(anova=T, weighted=F)

cat("Weighted, test multiple coefficients\n")
result4 <- perform.test(anova=T, weighted=T)

plot(result4$ltop$P.Value, result4$ftop$p.values,
     main="Weighted, multiple coefficients",
     xlab="Limma p values",
     ylab="Fitnoise p values")


#
#contrasts.fit documentation states:
#
#  Warning. For efficiency reasons, this function does not
#  re-factorize the design matrix for each probe. A consequence is
#  that, if the design matrix is non-orthogonal and the original fit
#  included quality weights or missing values, then the unscaled
#  standard deviations produced by this function are approximate
#  rather than exact. The approximation is usually acceptable. The
#  results are always exact if the original fit was a oneway model.
#



