import numpy as np

N = 15000 # numbe data points
k = 15 # number features
sigma = 0.1 # noise standard deviation

X = np.random.rand(N,k)

w = np.random.rand(k)
eps = sigma * np.random.rand(N)
t = np.dot(w, X.T) + eps

np.save("lr-X.npy", X)
np.save("lr-t.npy", t)
np.save("lr-theta.npy", w)
