import numpy as np
import numpy.random as npr
from scipy import optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import pypmc
import csv
import sys
sys.path.append('..')
import flymc as ff
import data

# Set hyperparameters
stepsize = 0.02          # size of Metropolis-Hastings step in theta
th0 = 1.5               # scale of weights
N_steps = 3000
N_ess = 2000

# Performance keys
KEY_NUM_REJECTS = "KEY_NUM_REJECTS"
KEY_ACCEPTANCE = "KEY_ACCEPTANCE"
KEY_LIK_EVALS = "KEY_LIK_EVALS"
KEY_NEG_LOG = "KEY_NEG_LOG"

def run_model(model, q=0.1, fly=False, verbose=False):
    '''
    function to run model
    '''
    th = np.random.randn(D) * th0

    # Init steppers
    if fly:
        z = ff.BrightnessVars(N, range(int(q*N)))
    else:
        z = ff.BrightnessVars(N, range(N))
    th_stepper = ff.ThetaStepMH(model.log_p_joint, stepsize)
    if fly:
        z_stepper = ff.zStepMH(model.log_pseudo_lik, q)

    # Trace list
    th_list = []
    num_rejects_list = []
    acceptance_list = []
    num_lik_evals_list = []
    neg_log_list = []
    performance_dict = {}

    # Run chain
    for _ in range(N_steps):
        num_lik_prev = model.num_lik_evals
        if _ % N_ess == 0 and _ > 0:
            #print pypmc.tools.convergence.ess(th_list) # TODO: is this correct?
            #print ess(th_list)
            np.savetxt('trace-{0}-{1}.csv'.format(model.name, _), np.array(th_list))
            th_list = []

        # Markov transition
        th = th_stepper.step(th, z)  # Markov transition step for theta
        if fly:
            z = z_stepper.step(th, z)  # Markov transition step for z
        th_list.append(th)

        # Record performance
        num_rejects = th_stepper.num_rejects
        num_lik_evals = model.num_lik_evals - num_lik_prev
        acceptance = 1.0 - sum(num_rejects_list)/float(_+1)
        neg_log = -1.0 * model.log_p_marg(th, increment_ctr=False)

        num_rejects_list.append(num_rejects)
        num_lik_evals_list.append(num_lik_evals)
        acceptance_list.append(acceptance)
        neg_log_list.append(neg_log)
 
        # Print info
        if verbose or (_ % 50 == 0):
            print "Accept rate: {0}".format(acceptance)
            print "Likelihood evals in iter {0}: {1}".format(_, num_lik_evals)
            print "Neg log posterior: {0}".format(neg_log)
            print "Number bright points: {0}".format(len(z.bright))

    performance_dict[KEY_NUM_REJECTS] = num_rejects_list
    performance_dict[KEY_LIK_EVALS] = num_lik_evals_list
    performance_dict[KEY_ACCEPTANCE] = acceptance_list
    performance_dict[KEY_NEG_LOG] = neg_log_list
    return performance_dict

def ess(th_list):
    th = np.array(th_list)
    th_mean = np.mean(th, axis=0)

    def autocorr(x, t):
        return np.mean((x[0:len(x)-t,:] - th_mean) * (x[t:len(x),:] - th_mean))

    return 1.0 * th.shape[0] / (1.0 + 2.0 * sum(map(lambda t: autocorr(th,t), range(1,th.shape[0]))))

def main():
    global N
    global D

    x, t = data.load_logistic_data()
    print x.shape, t.shape
    N, D = x.shape
    y0 = 1.5 # \xce in paper

    # model_mcmc = ff.LogisticModel(x, t, th0=th0, y0=y0)
    # print model_mcmc.num_lik_evals
    # th_mcmc, num_lik_evals_list_mcmc, th_list_mcmc = run_model(model_mcmc)
    # print model_mcmc.num_lik_evals

    model_untuned_flymc = ff.LogisticModel(x, t, th0=th0, y0=y0, name="untuned_flymc")
    # print model_flymc.num_lik_evals
    untuned_flymc_performance = run_model(model_untuned_flymc, q=0.1, fly=True) # q = prob(dim -> bright)
    # print model_flymc.num_lik_evals

    #_model = ff.LogisticModel(x, t, th0=th0)
    #th = np.random.randn(D) * th0
    #th_map = optimize.minimize(lambda x: -1*_model.log_p_marg(x), th)
    #model_flymc_map = ff.LogisticModel(x, t, th0=th0, th_map=th_map.x)
    #print run_model(model_flymc_map, q=0.01, fly=True)
    #print model_flymc_map.num_lik_evals

    # output traces to .csv
    # print num_lik_evals_list_flymc
    # with open('num_lik_evals_list_mcmc', 'wb') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(num_lik_evals_list_flymc)

    # plt.plot(num_lik_evals_list_mcmc)
    num_lik_evals_list_flymc = untuned_flymc_performance[KEY_LIK_EVALS]
    plt.plot(num_lik_evals_list_flymc)
    plt.show()

if __name__ == "__main__":
    main()
