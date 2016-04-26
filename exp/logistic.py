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
N_steps = 3000
N_ess = 2000

def preprocess():

    mndata = MNIST('../examples')
    mndata.load_training()
    ss_idx = filter(lambda i: mndata.train_labels[i] in [7,9], range(len(mndata.train_images)))
    data_ss = np.array(mndata.train_images)[ss_idx,:]
    label_ss = np.array(mndata.train_labels)[ss_idx]
    pca = PCA(n_components=50)
    pca.fit(data_ss)
    x = pca.transform(data_ss)
    x = np.concatenate((x, np.ones((x.shape[0],1))),axis=1)
    t = label_ss == 7

    np.save('logistic_x.npy', x)
    np.save('logistic_t.npy', t)
    exit(1)

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
        th_stepper = ff.ThetaStepMH(model.log_p_joint, stepsize)
        if fly:
            z__stepper = ff.zStepMH(model.log_pseudo_lik, q)
        th_lists = []           # trace of th
        num_rejects_list = []   #
        num_iter_list = []      # number of num_lik_evals for each iteration
        for _ in range(N_steps):
            num_lik_prev = model.num_lik_evals
            if _ % N_ess == 0 and _ > 0:
                #print pypmc.tools.convergence.ess(th_lists) # TODO: is this correct?
                #print ess(th_lists)
                np.savetxt('trace-untuned-{0}.csv'.format(_), np.array(th_lists))
                th_lists = []
            th = th_stepper.step(th, z)  # Markov transition step for theta


            if fly:
                z  = z__stepper.step(th ,z)  # Markov transition step for z
            th_lists.append(th)
            num_rejects_list.append(th_stepper.num_rejects)
            num_iter_list.append(model.num_lik_evals - num_lik_prev)
            print "Accept rate: {0}".format(1.0 - sum(num_rejects_list)/float(_+1))
            print "Likelihood evals in iter {0}: {1}".format(_, model.num_lik_evals - num_lik_prev)
            print "Neg log posterior: {0}".format(-1.0 * model.log_p_marg(th, increment_ctr=False))
            print "Number bright points: {0}".format(len(z.bright))

        return th, num_iter_list, th_lists
    # preprocess data and save to .npy file
    # so tha we dont have to PCA them each time
    # preprocess()

    x = np.load('logistic_x.npy')
    t = np.load('logistic_t.npy')
    print x.shape, t.shape
    N, D = x.shape
    y0 = 1.5 # \xce in paper


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


    # model_mcmc = ff.LogisticModel(x, t, th0=th0, y0=y0)
    # print model_mcmc.num_lik_evals
    # th_mcmc, num_iter_list_mcmc, th_lists_mcmc = run_model(model_mcmc)
    # print model_mcmc.num_lik_evals


    model_flymc = ff.LogisticModel(x, t, th0=th0, y0=y0)
    # print model_flymc.num_lik_evals
    th_flymc, num_iter_list_flymc, th_lists_flymc = run_model(model_flymc, q=0.1, fly=True) # q = prob(dim -> bright)
    # print model_flymc.num_lik_evals



    #_model = ff.LogisticModel(x, t, th0=th0)
    #th = np.random.randn(D) * th0
    #th_map = optimize.minimize(lambda x: -1*_model.log_p_marg(x), th)
    #model_flymc_map = ff.LogisticModel(x, t, th0=th0, th_map=th_map.x)
    #print run_model(model_flymc_map, q=0.01, fly=True)
    #print model_flymc_map.num_lik_evals

    # output traces to .csv
    # print num_iter_list_flymc
    # with open('num_iter_list_mcmc', 'wb') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(num_iter_list_flymc)

    # plt.plot(num_iter_list_mcmc)
    plt.plot(num_iter_list_flymc)
    plt.show()

if __name__ == "__main__":
    main()
