import numpy as np
import numpy.random as npr
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import flymc as ff

import load_data

# Set hyperparameters
N = 20                  # number of data points
D = 3                   # dimension of data points (plotting the data requires D=3)
stepsize = 0.75         # size of Metropolis-Hastings step in theta
th0 = 2.5               # scale of weights
y0 = 2                  # point at which to make bounds tight
q = 0.05                # Metropolis-Hastings proposal probability for z

def main():
    # Get softmax data
    x, t = load_data.get_softmax_data()
    K = 3
    print x.shape, t.shape
    # model = ff.MulticlassLogisticModel(x, t, K)
    exit(1)

    # Obtain joint distributions over z and th
    model = ff.LogisticModel(x, t, th0=th0, y0=y0)

    # Set up step functions
    th = np.random.randn(D) * th0
    z = ff.BrightnessVars(N)
    th_stepper = ff.ThetaStepMH(model.log_p_joint, stepsize)
    z__stepper = ff.zStepMH(model.log_pseudo_lik, q)

    plt.ion()
    ax = plt.figure(figsize=(8, 6)).add_subplot(111)
    while True:
        th = th_stepper.step(th, z)  # Markov transition step for theta
        z  = z__stepper.step(th ,z)  # Markov transition step for z

if __name__ == "__main__":
    main()
