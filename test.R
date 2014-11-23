
library(fitnoise)

a <- c(pi,2,3)
pyset("a",a)
pyexec("print a")
b <- pyget("a")

print(a == b)

pyexec("c = numpy.array([[1,2,3],[4,5,6]])")
print(pyget("c"))

pyset_single("d", "hello")
pyexec("print repr(d)")
print(pyget("d"))


#pyset("y", exp(matrix(runif(4),ncol=2)*10))
#pyexec("print y")
#pyexec("fit = Transform_to_t().fit(y)")
