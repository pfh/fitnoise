
source("backyard/synthetic.R")


result <- data.frame()

for(i in 1:1000) {
    print(i)
    data <- synth.elist(df=10.0, de_amount=5.0)
    
    data.l <- run.limma(data, 0.01)
    data.f <- run.fitnoise(data, 0.01)
    
    result[i,"limma.df"] <- data.l$fit$df.prior
    result[i,"limma.var"] <- data.l$fit$s2.prior
    result[i,"limma.sensitivity"] <- data.l$sensitivity
    result[i,"limma.fdr"] <- data.l$fdr
    
    result[i,"fitnoise.df"] <- data.f$fit$param$df
    result[i,"fitnoise.var"] <- data.f$fit$param$variance
    result[i,"fitnoise.sensitivity"] <- data.f$sensitivity
    result[i,"fitnoise.fdr"] <- data.f$fdr

    write.csv(result, "backyard/lorne-2015/out-synthetic.csv", row.names=F)
}
