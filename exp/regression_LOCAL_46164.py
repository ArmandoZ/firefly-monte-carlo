import numpy as np
import numpy.random as npr
from scipy import optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import pypmc
from mnist import MNIST
from sklearn.decomposition import PCA
import csv
import sys
sys.path.append('..')
import flymc as ff

# Set hyperparameters
stepsize = 0.02          # size of Metropolis-Hastings step in theta
th0 = 1.5               # scale of weights
N_steps = 2000
N_ess = 2000

def main():
    def run_model(model, q=0.1, fly=False):
        '''
        function to run model
        '''
        th = np.random.randn(D) * th0
        if fly:
            z = ff.BrightnessVars(N, range(int(q*N)))
        else:
            z = ff.BrightnessVars(N, range(N))
        th_stepper = ff.ThetaStepSlice(model.log_p_joint, stepsize)
        if fly:
            z__stepper = ff.zStepMH(model.log_pseudo_lik, q)
        th_lists = []           # trace of th
        num_rejects_list = []   #
        num_iter_list = []      # number of num_lik_evals for each iteration
        neg_log_post_list = []  # neg log posterior
        for _ in range(N_steps):
            num_lik_prev = model.num_lik_evals
            th = th_stepper.step(th, z)  # Markov transition step for theta
            if fly:
                z  = z__stepper.step(th ,z)  # Markov transition step for z
            th_lists.append(th)
            num_rejects_list.append(th_stepper.num_rejects)
            num_iter_list.append(model.num_lik_evals - num_lik_prev)
            neg_log_post_list.append(-1.0 * model.log_p_marg(th, increment_ctr=False))
            print "Accept rate: {0}".format(1.0 - sum(num_rejects_list)/float(_+1))
            print "Likelihood evals in iter {0}: {1}".format(_, num_iter_list[-1])
            print "Neg log posterior: {0}".format(neg_log_post_list[-1])
            print "Number bright points: {0}".format(len(z.bright))

        return num_iter_list, th_lists, neg_log_post_list

    # preprocess data and save to .npy file
    # so tha we dont have to PCA them each time
    # preprocess()

    x = np.load('lr-X.npy')
    t = np.load('lr-t.npy')
    print x.shape, t.shape
    N, D = x.shape
    y0 = 1.5 # \xce in paper

    model_mcmc = ff.RobustRegressionModel(x, t, th0=th0, y0=y0)
    num_iter_mcmc, th_mcmc, neg_log_post_mcmc = run_model(model_mcmc)
    np.savetxt('regression_num_iter_mcmc.csv', np.array(num_iter_mcmc))
    np.savetxt('regression_th_mcmc.csv', np.array(th_mcmc))
    np.savetxt('regression_neg_log_post_mcmc.csv', np.array(neg_log_post_mcmc))

    model_flymc = ff.RobustRegressionModel(x, t, th0=th0, y0=y0)
    num_iter_flymc, th_flymc, neg_log_post_flymc = run_model(model_flymc, q=0.1, fly=True) # q = prob(dim -> bright)
    np.savetxt('regression_num_iter_flymc.csv', np.array(num_iter_flymc))
    np.savetxt('regression_th_flymc.csv', np.array(th_flymc))
    np.savetxt('regression_neg_log_post_flymc.csv', np.array(neg_log_post_flymc))

    # _model = ff.RobustRegressionModel(x, t, th0=th0)    # dummy model used to optimize th
    # th = np.random.randn(D) * th0
    # th_map = optimize.minimize(lambda x: -1*_model.log_p_marg(x), th)
    # np.save('regression_th_map.npy', th_map.x)
    th_map = np.load('regression_th_map.npy')
    model_flymc_map = ff.RobustRegressionModel(x, t, th0=th0, th_map=th_map)
    num_iter_flymc_map, th_flymc_map, neg_log_post_flymc_map = run_model(model_flymc_map, q=0.01, fly=True)

    # output traces to .csv


    np.savetxt('regression_num_iter_flymc_map.csv', np.array(num_iter_flymc_map))
    np.savetxt('regression_th_flymc_map.csv', np.array(th_flymc_map))
    np.savetxt('regression_neg_log_post_flymc_map.csv', np.array(neg_log_post_flymc_map))

if __name__ == "__main__":
    main()
