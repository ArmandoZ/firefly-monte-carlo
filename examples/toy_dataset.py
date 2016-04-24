import numpy as np
import numpy.random as npr
from scipy import optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import pypmc
from mnist import MNIST
from sklearn.decomposition import PCA


import sys
sys.path.append('..')
import flymc as ff

# Set hyperparameters
stepsize = 0.05          # size of Metropolis-Hastings step in theta
th0 = 0.20               # scale of weights

N_steps = 3000
N_ess = 2000

# Cosmetic settings
mpl.rcParams['axes.linewidth'] = 3
mpl.rcParams['lines.linewidth'] = 7
mpl.rcParams['toolbar'] = "None"
mpl.rcParams['figure.facecolor'] = "1"

def main():
    mndata = MNIST('.')
    mndata.load_training()
    #mndata.load_testing()

    ss_idx = filter(lambda i: mndata.train_labels[i] in [7,9], range(len(mndata.train_images)))
    data_ss = np.array(mndata.train_images)[ss_idx,:]
    label_ss = np.array(mndata.train_labels)[ss_idx]

    pca = PCA(n_components=10) # TODO: change to 50
    pca.fit(data_ss)
    x = pca.transform(data_ss)
    x = np.concatenate((x, np.ones((x.shape[0],1))),axis=1)
    t = label_ss == 7

    N, D = x.shape
    y0 = 1.5 # \xce in paper

    # Generate synthetic data
    # x = 2 * npr.rand(N,D) - 1  # data features, an (N,D) array
    # x[:, 0] = 1
    # th_true = 10.0 * np.array([0, 1, 1])
    # y = np.dot(x, th_true[:, None])[:, 0]
    # t = npr.rand(N) > (1 / ( 1 + np.exp(y)))  # data targets, an (N) array of 0s and 1s

    # Obtain joint distributions over z and th

    # Set up step functions
    def run_model(model, q=0.1, fly=False):
        th = np.random.randn(D) * th0
        if fly:
            z = ff.BrightnessVars(N)
        else:
            z = ff.BrightnessVars(N, range(N))
        th_stepper = ff.ThetaStepMH(model.log_p_joint, stepsize)
        if fly:
            z__stepper = ff.zStepMH(model.log_pseudo_lik, q)
        ths = []
        for _ in range(N_steps):
            num_lik_prev = model.num_lik_evals
            if _ % N_ess == 0 and _ > 0:
                #print pypmc.tools.convergence.ess(ths) # TODO: is this correct?
                #print ess(ths)
                np.savetxt('trace-untuned-{0}.csv'.format(_), np.array(ths))
                ths = []
            th = th_stepper.step(th, z)  # Markov transition step for theta
            if fly:
                z  = z__stepper.step(th ,z)  # Markov transition step for z
            ths.append(th)
            print "Likelihood evals in iter {0}: {1}".format(_, model.num_lik_evals - num_lik_prev)
            print "Number bright points: {0}".format(len(z.bright))
        return th

    def ess(th_list):
        th = np.array(th_list)
        th_mean = np.mean(th, axis=0)
        def autocorr(x, t):
            return np.mean((x[0:len(x)-t,:] - th_mean) * (x[t:len(x),:] - th_mean))
        return 1.0 * th.shape[0] / (1.0 + 2.0 * sum(map(lambda t: autocorr(th,t), range(1,th.shape[0]))))

    # print ess([
    #     np.array([1]),
    #     np.array([1.1]),
    #     np.array([0.9]),
    #     np.array([1])
    #     ])


    model_mcmc = ff.LogisticModel(x, t, th0=th0, y0=y0)
    #print model_mcmc.num_lik_evals
    #print run_model(model_mcmc)
    #print model_mcmc.num_lik_evals

    model_flymc = ff.LogisticModel(x, t, th0=th0, y0=y0)
    #print model_flymc.num_lik_evals
    print run_model(model_flymc, q=0.1, fly=True) # q = prob(dim -> bright)
    #print model_flymc.num_lik_evals

    #_model = ff.LogisticModel(x, t, th0=th0)
    #th = np.random.randn(D) * th0
    #th_map = optimize.minimize(lambda x: -1*_model.log_p_marg(x), th)
    #model_flymc_map = ff.LogisticModel(x, t, th0=th0, th_map=th_map.x)
    #print run_model(model_flymc_map, q=0.01, fly=True)
    #print model_flymc_map.num_lik_evals

    # plt.ion()
    # ax = plt.figure(figsize=(8, 6)).add_subplot(111)
    # while True:
    #     th = th_stepper.step(th, z)  # Markov transition step for theta
    #     z  = z__stepper.step(th ,z)  # Markov transition step for z
    #     update_fig(ax, x, y, z, th, t)
    #     plt.draw()
    #     plt.pause(0.05)

def update_fig(ax, x, y, z, th, t):
    b = np.zeros(N)
    b[z.bright] = 1

    bright1s = (   t  *    b ).astype(bool)
    bright0s = ((1-t) *    b ).astype(bool)
    dark1s   = (   t  * (1-b)).astype(bool)
    dark0s   = ((1-t) * (1-b)).astype(bool)
    ms, bms, mew = 45, 45, 5

    ax.clear()
    ax.plot(x[dark0s,1],   x[dark0s,2],  's', mec='Blue', mfc='None', ms=ms,  mew=mew)
    ax.plot(x[dark1s,1],   x[dark1s,2],  'o', mec='Red',  mfc='None', ms=ms,  mew=mew)
    ax.plot(x[bright0s,1], x[bright0s,2],'s', mec='Blue', mfc='Blue', ms=bms, mew=mew)
    ax.plot(x[bright1s,1], x[bright1s,2],'o', mec='Red',  mfc='Red',  ms=bms, mew=mew)

    X = np.arange(-3,3)
    th1, th2, th3 = th[0], th[1], th[2]
    Y = (-th1 - th2 * X) / th3

    ax.plot(X, Y, color='grey')
    lim = 1.15
    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])
    ax.set_yticks([])
    ax.set_xticks([])

if __name__ == "__main__":
    main()
