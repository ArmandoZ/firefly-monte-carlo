from numpy import random as npr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def f(x):
    return (3 + npr.rand()) * x + 5 + npr.rand()
print [f(x) for x in range(1000)]
