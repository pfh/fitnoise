

library(unmodlimma)

set.seed(2015)

n <- 100
m <- 6

test.coef <- c(2,3)
null.coef <- c(1)

X <- rbind(
    c(1,0,0),
    c(1,0,0),
    c(1,1,0),
    c(1,1,0),
    c(1,0,1),
    c(1,0,1)
    )
nullX <- X[,null.coef,drop=F]

y <- matrix(rnorm(n*m), nrow=n)

weights <- matrix(runif(n*m), nrow=n)

# Setting weights to NULL removes the discrepancy
#weights <- NULL

fit <- lmFit(y,X,weights=weights)
fit <- eBayes(fit)
top <- topTable(fit, coef=test.coef, sort="none", number=n)

anova.p <- c()
for(i in 1:n) {
   an <- anova(
       lm(y[i,] ~ X+0,weights=weights[i,]), 
       lm(y[i,] ~ nullX+0,weights=weights[i,])
       )
   anova.p[i] <- an$"Pr(>F)"[2]
}

plot(anova.p, top$P.Value,
     xlab="F-test on nested linear models p value",
     ylab="unmodlimma p value"
     )

cat("\nMax discrepancy in p values",
    max(abs(top$P.Value - anova.p)),
    "\n")
