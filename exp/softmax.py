import numpy as np
import numpy.random as npr
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import sys
sys.path.append('..')
import flymc as ff

import load_data

# Set hyperparameters
y0 = 2                  # point at which to make bounds tight
q = 0.01                # Metropolis-Hastings proposal probability for z
stepsize = 0.07          # size of Metropolis-Hastings step in theta
th0 = 0.20               # scale of weights

N_steps = 3000
N_ess = 2000

def preprocess():
    x, t = load_data.get_softmax_data()
    pca = PCA(n_components=256)
    pca.fit(x)
    x = pca.transform(x)
    np.save('softmax_x.npy', x)
    np.save('softmax_t.npy', t)
    exit(1)

def main():
    def run_model(model, q=0.1, fly=False):
        th = np.random.randn(K, D) * th0
        if fly:
            z = ff.BrightnessVars(N)
        else:
            z = ff.BrightnessVars(N, range(N))
        th_stepper = ff.ThetaStepLangevin(model.log_p_joint, model.D_log_p_joint, stepsize)
        if fly:
            z__stepper = ff.zStepMH(model.log_pseudo_lik, q)
        ths = []
        for _ in range(N_steps):
            num_lik_prev = model.num_lik_evals
            if _ % N_ess == 0 and _ > 0:
                #print pypmc.tools.convergence.ess(ths) # TODO: is this correct?
                #print ess(ths)
                np.savetxt('softmax-trace-untuned-{0}.csv'.format(_), np.array(ths))
                ths = []
            th = th_stepper.step(th, z)  # Markov transition step for theta
            if fly:
                z  = z__stepper.step(th ,z)  # Markov transition step for z
            ths.append(th)
            print "Likelihood evals in iter {0}: {1}".format(_, model.num_lik_evals - num_lik_prev)
            print "Number bright points: {0}".format(len(z.bright))
        return th
    # preprocess data and save to .npy file
    # so tha we dont have to PCA them each time
    # preprocess()

    # read softmax data
    x = np.load('softmax_x.npy')
    t = np.load('softmax_t.npy')
    K = 3
    N, D = x.shape

    print x.shape, t.shape


    # model_mcmc = ff.MulticlassLogisticModel(x, t, K, th0=th0, y0=y0)
    # print model_mcmc.num_lik_evals
    # print run_model(model_mcmc)
    # print model_mcmc.num_lik_evals

    model_flymc = ff.MulticlassLogisticModel(x, t, K, th0, y0)
    print run_model(model_flymc, q=0.1, fly=True)

if __name__ == "__main__":
    main()
