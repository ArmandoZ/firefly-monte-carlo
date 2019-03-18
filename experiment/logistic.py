import numpy as np
import numpy.random as npr
from scipy import optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import pypmc
import csv
import sys
import os
sys.path.append('..')
import flymc as ff
import data

# Set hyperparameters
stepsize = 0.02          # size of Metropolis-Hastings step in theta
th0 = 1.5               # scale of weights
N_steps = 3000
N_ess = 2000

# Performance keys
KEY_NAME = "KEY_NAME"
KEY_NUM_REJECTS = "KEY_NUM_REJECTS"
KEY_ACCEPTANCE = "KEY_ACCEPTANCE"
KEY_LIK_EVALS = "KEY_LIK_EVALS"
KEY_NEG_LOG = "KEY_NEG_LOG"
KEY_SAMPLE_EVALS = "KEY_SAMPLE_EVALS"
KEY_SAMPLE_ESS = "KEY_SAMPLE_ESS"

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
    num_rejects_list = []
    acceptance_list = []
    num_lik_evals_list = []
    neg_log_list = []
    performance_dict = {}
    # Trace - Sampling
    sample_num_lik_evals_list = []
    sample_th_list = []

    # Run chain
    for _ in range(N_steps + N_ess):
        num_lik_prev = model.num_lik_evals
        # Markov transition
        th = th_stepper.step(th, z)  # Markov transition step for theta
        if fly:
            z = z_stepper.step(th, z)  # Markov transition step for z

        # Record performance
        num_rejects = th_stepper.num_rejects
        num_lik_evals = model.num_lik_evals - num_lik_prev
        acceptance = 1.0 - sum(num_rejects_list)/float(_+1)
        neg_log = -1.0 * model.log_p_marg(th, increment_ctr=False)

        if _ < N_steps:
            num_rejects_list.append(num_rejects)
            num_lik_evals_list.append(num_lik_evals)
            acceptance_list.append(acceptance)
            neg_log_list.append(neg_log)
        else: 
            # _ >= N_steps
            sample_num_lik_evals_list.append(num_lik_evals)
            sample_th_list.append(th)
 
        # Print info
        if verbose or (_ % 50 == 0):
            print "Accept rate: {0}".format(acceptance)
            print "Likelihood evals in iter {0}: {1}".format(_, num_lik_evals)
            print "Neg log posterior: {0}".format(neg_log)
            print "Number bright points: {0}".format(len(z.bright))

    sample_ess = ess(sample_th_list)
    sample_evals = sum(sample_num_lik_evals_list)
    np.savetxt('trace-{0}-{1}.csv'.format(model.name, sample_ess), np.array(sample_th_list))

    performance_dict[KEY_NAME] = model.name
    performance_dict[KEY_NUM_REJECTS] = num_rejects_list
    performance_dict[KEY_LIK_EVALS] = num_lik_evals_list
    performance_dict[KEY_ACCEPTANCE] = acceptance_list
    performance_dict[KEY_NEG_LOG] = neg_log_list
    performance_dict[KEY_SAMPLE_EVALS] = sample_evals
    performance_dict[KEY_SAMPLE_ESS] = sample_ess

    return performance_dict

def ess(th_list):
    # Alternate: pypmc.tools.convergence.ess(th_list) # TODO: is this correct?
    th = np.array(th_list)
    th_mean = np.mean(th, axis=0)

    def autocorr(x, t):
        return np.mean((x[0:len(x)-t,:] - th_mean) * (x[t:len(x),:] - th_mean))

    return 1.0 * th.shape[0] / (1.0 + 2.0 * sum(map(lambda t: autocorr(th,t), range(1,th.shape[0]))))

def save_results(performance_results):
    ## Plot
    # Plot negative log posteriors
    if not os.path.exists("./result"):
        os.mkdir("./result")

    plt.cla()
    for performance_result in performance_results:
        y = performance_result[KEY_NEG_LOG]
        label = performance_result[KEY_NAME]
        plt.plot(range(len(y)), y, label=label)
    plt.title("Negative Log Posteriors")
    plt.legend(loc='best')
    plt.savefig("./result/%s.png" % "Negative Log Posteriors")

    # Plot Mean Likelihood Evaluations
    plt.cla()
    for performance_result in performance_results:
        y = performance_result[KEY_LIK_EVALS]
        label = performance_result[KEY_NAME]
        plt.plot(range(len(y)), y, label=label)
    plt.title("Mean Likelihood Evaluations")
    plt.legend(loc='best')
    plt.savefig("./result/%s.png" % "Mean Likelihood Evaluations")

    ## Sample
    with open("./result/sample_result.csv", 'w') as f:
        f.write("model_name,ESS,mean_likelihood_evaluation,product\n")
        for performance_result in performance_results:
            model_name = performance_result[KEY_NAME]
            sample_ess = performance_result[KEY_SAMPLE_ESS]
            sample_evals = performance_result[KEY_SAMPLE_EVALS]
            f.write("{0},{1},{2},{3}\n" % (model_name, sample_ess, sample_evals, sample_ess * sample_evals))

def main():
    global N
    global D

    x, t = data.load_logistic_data()
    print x.shape, t.shape
    N, D = x.shape
    y0 = 1.5 # \xce in paper

    performance_results = []

    model_mcmc = ff.LogisticModel(x, t, th0=th0, y0=y0, name="regular mcmc")
    mcmc_performance = run_model(model_mcmc)
    performance_results.append(mcmc_performance)

    model_untuned_flymc = ff.LogisticModel(x, t, th0=th0, y0=y0, name="untuned_flymc")
    untuned_flymc_performance = run_model(model_untuned_flymc, q=0.1, fly=True) # q = prob(dim -> bright)
    performance_results.append(untuned_flymc_performance)

    tmp_model = ff.LogisticModel(x, t, th0=th0)
    th_map = optimize.minimize(
            fun=lambda th: -1.0*tmp_model.log_p_marg(th),
            x0=np.random.randn(D)*th0,
            jac=lambda th: -1.0*tmp_model.D_log_p_marg(th),
            method='BFGS',
            options={
                'maxiter': 100,
                'disp': True
            })

    model_tuned_flymc = ff.LogisticModel(x, t, th0=th0, th_map=th_map.x, name="tuned_flymc")
    tuned_flymc_performance = run_model(model_tuned_flymc, q=0.01, fly=True)
    performance_results.append(tuned_flymc_performance)

    save_results(performance_results)

if __name__ == "__main__":
    main()
