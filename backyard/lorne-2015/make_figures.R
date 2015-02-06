
library(ggplot2)
library(reshape)
library(scales)

mytheme <- theme_minimal() #+ theme(panel.grid.minor.y = element_blank(), panel.grid.major.x = element_blank())


save <- function(name, item) {
    png(sprintf("backyard/lorne-2015/out-%s.png",name), width=3,height=3,units="in",res=300)
    print(item)
    dev.off()
}


info <- function(name, vec) {
    cat(sprintf("%s %.3f [%.3f,%.3f]\n",
        name,
        mean(vec),
        quantile(vec, 0.025),
        quantile(vec, 0.975)))
}

infob <- function(name, vec) {
    cat(sprintf("%s %.3f [%.3f,%.3f]\n",
        name,
        vec[1],
        quantile(vec, 0.025),
        quantile(vec, 0.975)))
}

synth <- read.csv("backyard/lorne-2015/out-synthetic.csv")

patboot <- read.csv("backyard/lorne-2015/out-patseq-bootstrap.csv")
#patboot <- patboot[1:200, ]
#print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

top <- read.csv("backyard/lorne-2015/out-patseq-top.csv")
top$sig <- top$q <= 0.01


sink("backyard/lorne-2015/out-report.txt", split=T)

info("limma.df", synth$limma.df)
info("limma.sd", sqrt(synth$limma.var))
info("limma.sensitivity", synth$limma.sensitivity)
info("limma.fdr", synth$limma.fdr)

info("fitnoise.df", synth$fitnoise.df)
info("fitnoise.sd", sqrt(synth$fitnoise.var))
info("fitnoise.sensitivity", synth$fitnoise.sensitivity)
info("fitnoise.fdr", synth$fitnoise.fdr)

print(nrow(patboot))

infob("patseq df", patboot$df)
infob("patseq read_sd", sqrt(patboot$read_variance))
infob("patseq sample_sd", sqrt(patboot$sample_variance))
infob("patseq significant", patboot$significant)
infob("patseq limma df", patboot$limma.df)
infob("patseq limma significant", patboot$limma.significant)


cat(sum(top$sig), "significant\n")
cat(sum(top$sig & top$X2 > 0.0), "significant and positive change\n")
cat(mean(top$X2[top$sig]), "mean change\n")

sink()

g <- geom_violin()

save("synth-df", ggplot(
    melt(data.frame(Limma=log10(synth$limma.df),Fitnoise=log10(synth$fitnoise.df))),
      aes(x=variable,y=value))
    + mytheme
    + g
    + xlab("")
    + ylab("prior degrees of freedom v")
    + scale_y_continuous(limits=c(0,3), breaks=c(0,1,2,3), labels = function(x) 10**x)
    + annotation_logticks(sides="l")
    )


save("synth-variance", ggplot(
    melt(data.frame(Limma=synth$limma.var,Fitnoise=synth$fitnoise.var)),
      aes(x=variable,y=value))
    + mytheme
    + g
    + xlab("")
    + ylab(expression(paste("prior variance ", σ^2)))
    )


save("synth-sensitivity", ggplot(
    melt(data.frame(Limma=synth$limma.sensitivity,Fitnoise=synth$fitnoise.sensitivity)),
      aes(x=variable,y=value))
    + mytheme
    + g
    + xlab("")
    + ylab("sensitivity")
    + scale_y_continuous(labels=function(x) sprintf("%.0f%%", x*100))
    )


save("synth-fdr", ggplot(
    melt(data.frame(Limma=synth$limma.fdr,Fitnoise=synth$fitnoise.fdr)),
      aes(x=variable,y=value))
    + mytheme
    + g
    + xlab("")
    + ylab("False Discovery Rate")
    )


top <- read.csv("backyard/lorne-2015/out-patseq-top.csv")
top$sig <- top$q <= 0.01

save("ma", ggplot(
    top, aes(x=average, y=X2 ))
    + mytheme
    + geom_point(size=0.5)
    + geom_point(data=subset(top,sig), color="#ff0000", size=1.0)
    + xlab("Average tail length")
    + ylab("Difference in tail lengths")
    )    


save("vsdepth", ggplot(
    top, aes(x=log10(total.count), y=X2 ))
    + mytheme
    + geom_point(size=0.5)
    + geom_point(data=subset(top,sig), color="#ff0000", size=1.0)
    + scale_x_continuous(limits=c(1,7), breaks=c(2,4,6), labels = function(x) sprintf("%d",10**x))
    + annotation_logticks(sides="b")
    + xlab("Total reads")
    + ylab("Difference in tail lengths")
    )    


