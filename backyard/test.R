
source("backyard/synthetic.R")

data <- synth.elist(de_amount=10.0)
datar <- run.limma(data, 0.01)

print(datar$sensitivity)
print(datar$fdr)

print(datar$fit$df.prior)

dataf <- run.fitnoise(data, 0.01)
cat(dataf$fit$description,"\n")
print(dataf$fit$param)
print(dataf$sensitivity)
print(dataf$fdr)

for(i in seq_len(10)) {
    data2 <- bootstrap(data)
    
    data2r <- run.limma(data2, 0.01)
    print(data2r$fit$df.prior)
    
    data2f <- run.fitnoise(data2, 0.01)
    print(data2f$fit$param$df)
    
    cat("\n")
}