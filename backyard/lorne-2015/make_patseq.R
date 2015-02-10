
arg <- commandArgs(T)

if (arg == "1") {
  model <- "Model_t_patseq_v1()"
  name <- "patseq1a"
} else if (arg == "2") {
  model <- "Model_t_patseq_v2()"
  name <- "patseq2a"
} else if (arg == "3") {
  model <- "Model_t_patseq_v3()"
  name <- "patseq3a"
} else {
  stop("Bad arg")
}

source("backyard/synthetic.R")

library(nesoni)

data <- read.grouped.table("backyard/lorne-2015/counts.csv")

#data$Tail[ data$Count == 0 ] <- 0.0

columns <- c(7,8,9,4,5,6)

myelist <- new("EList",list(
    E = as.matrix(data$Tail[,columns]),
    other = list(counts=as.matrix(data$Tail_count[,columns])),
    genes = data$Annotation
    ))

#good <- rowSums(myelist$other$counts) >= 10
#cat("Selecting", sum(good), "of", length(good), "peaks")
#myelist <- myelist[good,]



design <- cbind(c(1,1,1,1,1,1),c(0,0,0,1,1,1))

pack <- list(elist=myelist, design=design, test_coef=2)

fit <- run.fitnoise(pack, 0.01, model=model)

print(fit$fit$description)
print(fit$fit$param)

print(sum(fit$declared_de))


pack$elist$weights <- fitnoise.weights(fit$fit)

fitlimma <- run.limma(pack, 0.01)
print(sum(fitlimma$declared_de))

print(names(fitlimma$fit))

toptab <- data.frame(
    q=fit$fit$q.values,
    p=fit$fit$p.values,
    limma.q=fitlimma$toptable$adj.P.Val,
    average=fit$fit$averages,
    fit$fit$coef,
    total.count = rowSums(fit$elist$other$counts),
    fit$elist$genes
    )

write.csv(toptab[ order(toptab$p), ], 
    sprintf("backyard/lorne-2015/out-%s-top.csv", name), row.names=F)


result <- data.frame()

for(i in 1:1000) {
    if (i == 1)
        bootpack <- pack
    else
        bootpack <- bootstrap(pack)
    bootfit <- run.fitnoise(bootpack, 0.01, model=model)

    result[i, "df"] <- bootfit$fit$param$df
    result[i, "read_variance"] <- bootfit$fit$param$read_variance
    result[i, "sample_variance"] <- bootfit$fit$param$sample_variance
    result[i, "significant"] <- sum(bootfit$declared_de)
    result[i, "score"] <- bootfit$fit$score
    result[i, "noise_combined_p_value"] <- bootfit$fit$noise.combined.p.value

    bootlimma <- run.limma(bootpack, 0.01)
    result[i,"limma.df"] <- bootlimma$fit$df.prior
    result[i,"limma.var"] <- bootlimma$fit$s2.prior
    result[i,"limma.significant"] <- sum(bootlimma$declared_de)

    write.csv(result, 
        sprintf("backyard/lorne-2015/out-%s-bootstrap.csv",name), row.names=F)
}
