##install.packages("coda")
library(coda)

trace <- read.csv('./examples/trace-normal-1000.csv', sep=' ',header=F)
trace <- read.csv('./examples/trace-1000.csv', sep=' ',header=F)
trace <- read.csv('./examples/trace-untuned-1000.csv', sep=' ',header=F)
str(trace)

trace$V10

ESS <- effectiveSize(trace)
ESS
mean(ESS)

# ESS normal
# V1        V2        V3        V4        V5        V6        V7        V8        V9       V10       V11 
# 5.006438  5.901716  2.684581  7.671832 64.743845 43.509831 55.889921 46.678764 10.331104 19.603918  8.738540 

# ESS MAP
# V1         V2         V3         V4         V5         V6         V7         V8         V9        V10        V11 
# 11.136354   6.477447  37.978618   8.288035 154.772838  19.173697  15.363953  27.703271  30.878931   8.831721  24.187820 