
# Check if fitnoise with forced hyperparameter choices exactly matches unmoderated ANOVA.

library(fitnoise)

set.seed(2015)

n <- 100
m <- 6

c <- 3
test.coef <- c(1,2)
null.coef <- c(3)

X <- matrix(rnorm(m*c), nrow=m)
y <- matrix(rnorm(n*m), nrow=n)
weights <- matrix(runif(n*m), nrow=n)

ffit <- fitnoise.fit(
    y, X, weights=weights, 
    force.param=c(0.0, 1.0)
    )

ftop <- fitnoise.test(ffit, coef=test.coef)

anova.p <- c()
for(i in 1:n) {
   an <- anova(
       lm(y[i,] ~ X+0,weights=weights[i,]), 
       lm(y[i,] ~ X[,null.coef,drop=F]+0,weights=weights[i,])
       )
   anova.p[i] <- an$"Pr(>F)"[2]
}

plot(anova.p, ftop$p.values,
     xlab="ANOVA nested linear models p value",
     ylab="Fitnoise p value"
     )

cat("\nMax discrepancy in p values",
    max(abs(ftop$p.values - anova.p)),
    "\n")
