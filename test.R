
library(fitnoise)

#a <- c(pi,2,3)
#pyset("a",a)
#pyexec("print a")
#b <- pyget("a")
#
#print(a == b)
#
#pyexec("c = numpy.array([[1,2,3],[4,5,6]])")
#print(pyget("c"))
#
#pyset_single("d", "hello")
#pyexec("print repr(d)")
#print(pyget("d"))


#pyset("y", exp(matrix(runif(4),ncol=2)*10))
#pyexec("print y")
#pyexec("fit = Transform_to_t().fit(y)")


n <- 1000
m <- 4
x <- 2 ** (1.0*matrix(runif(n*m), nrow=n) + 10.0*runif(n))

f <- fitnoise.transform(x)

print(f$fit)

plot(f$y[,1],f$y[,2])
