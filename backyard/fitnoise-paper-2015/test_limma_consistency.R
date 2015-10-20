
library("limma")

library("NBPSeq")
data("arab")
counts <- arab
counts <- counts[rowSums(counts) >= 1, ]
group <- factor(c("a","a","a","b","b","b"))
batch <- factor(c("b1","b2","b3","b1","b2","b3"))
design1 <- model.matrix(~ batch + group)
design2 <- model.matrix(~ 0 + batch + group)

elist <- voom(counts, design=design1)

fit1 <- lmFit(elist, design=design1)
fit1 <- eBayes(fit1)
print( topTable(fit1, coef=c(2,3)) )

fit2 <- lmFit(elist, design=design2)
fit2 <- eBayes(fit2)
contrasts <- cbind( c(-1,1,0,0), c(-1,0,1,0) )
fit2c <- contrasts.fit(fit2, contrasts)
print( topTable(fit2c, coef=c(1,2)) )