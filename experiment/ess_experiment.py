# coding=utf-8
import numpy as np
import numpy.random as npr
import pypmc
import csv
import os
from multiESS import multiESS

def ess_pypmc(th_list):
    return pypmc.tools.convergence.ess(th_list)

def ess_multi(th_list):
    return multiESS(th_list)

def ess(th_list):
    th = np.array(th_list)
    th_mean = np.mean(th, axis=0)

    def autocorr(x, t):
        return np.mean((x[0:len(x)-t,:] - th_mean) * (x[t:len(x),:] - th_mean))

    return 1.0 * th.shape[0] / (1.0 + 2.0 * sum(map(lambda t: autocorr(th,t), range(1,th.shape[0]))))

def read_sample_from_csv(fileName):
    if fileName == "daily-minimum-temperatures-in-me.csv":
        temperature_sequence = []
        csv_file = csv.reader(open("daily-minimum-temperatures-in-me.csv", 'r'))
        for i, temp in enumerate(csv_file):
            if i > 0 and i < 3651:
                temperature_sequence.append(float(temp[1]))
        temperature_sequence = np.array(temperature_sequence)
        return temperature_sequence[:, np.newaxis]
    else:
        tmp = np.loadtxt(fileName, dtype=np.str, delimiter=" ")
        data = tmp[0:].astype(np.float64)
        return data

def main():
    fileNames = []
    fileNames.append("trace-regular mcmc--557.028807796.csv")
    fileNames.append("trace-tuned_flymc-7035.9984602.csv")
    fileNames.append("trace-untuned_flymc--5939.88475791.csv")
    fileNames.append("daily-minimum-temperatures-in-me.csv")
    for fileName in fileNames:
        print fileName
        samples = read_sample_from_csv(fileName)
        print samples.shape
        print "ess_pypmc %.4f" % ess_pypmc(samples)
        print "ess_multi %.4f" % ess_multi(samples)
        print "ess       %.4f" % ess(samples)
        print 

if __name__ == "__main__":
    main()
