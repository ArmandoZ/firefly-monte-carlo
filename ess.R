#install.packages("coda")
library(coda)

# 5.01
trace <- read.csv('./experiment/trace-regular mcmc-2000.csv', sep=' ',header=F)
ESS <- effectiveSize(trace)
mean(ESS)
# 4.71
trace <- read.csv('./experiment/trace-untuned_flymc-2000.csv', sep=' ',header=F)
ESS <- effectiveSize(trace)
mean(ESS)
# 5.07
trace <- read.csv('./experiment/trace-tuned_flymc-2000.csv', sep=' ',header=F)
ESS <- effectiveSize(trace)
mean(ESS)
