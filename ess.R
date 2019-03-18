#install.packages("coda")
library(coda)

trace <- read.csv('./experiment/trace-regular mcmc--557.028807796.csv', sep=' ',header=F)
trace <- read.csv('./experiment/trace-untuned_flymc-20000.csv', sep=' ',header=F)
trace <- read.csv('./experiment/trace-untuned_flymc--5939.88475791.csv', sep=' ',header=F)
str(trace)

trace$V10

ESS <- effectiveSize(trace)
mean(ESS)
