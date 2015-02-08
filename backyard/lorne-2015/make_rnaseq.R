
{

arg <- commandArgs(T)

if (arg == "n") {
    model <- "Model_normal()"
    name <- "rnaseqn"
} else if (arg == "t") {
    model <- "Model_t()"
    name <- "rnaseqt"
} else if (arg == "nps") {
    model <- "Model_normal_per_sample()"
    name <- "rnaseqnps"
} else if (arg == "tps") {
    model <- "Model_t_per_sample()"
} else if (arg == "nf1") {
    model <- "Model_normal_factors(1)"
} else if (arg == "tf1") {
    model <- "Model_t_factors(1)"
} else {
    stop("Bad arg")
}



source("backyard/synthetic.R")

library(nesoni)

data <- read.grouped.table("backyard/lorne-2015/counts.csv")

columns <- c(7,8,9,4,5,6)

design <- cbind(c(1,1,1,1,1,1),c(0,0,0,1,1,1))

myelist <- voom(data$Count[,columns], design=design)

pack <- list(elist=myelist, design=design, test_coef=2)

fit <- run.fitnoise(pack, 0.01, model=model)

print(fit$fit$description)
print(fit$fit$param)

print(sum(fit$declared_de))


fitlimma <- run.limma(pack, 0.01)
print(fitlimma$fit$df.prior)
print(fitlimma$fit$s2.prior)
print(sum(fitlimma$declared_de))


}