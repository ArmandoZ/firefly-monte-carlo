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
from sklearn.preprocessing import normalize

# Set hyperparameters
stepsize = 0.06          # size of Metropolis-Hastings step in theta
th0 = 0.2               # scale of weights
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
        '''
        function to run model
        '''
        th = np.random.randn(K, D) * th0
        if fly:
            z = ff.BrightnessVars(N, range(int(q*N)))
        else:
            z = ff.BrightnessVars(N, range(N))
        th_stepper = ff.ThetaStepLangevin(model.log_p_joint, model.D_log_p_joint, stepsize)
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
            th_lists.append(np.reshape(th, [K*D,]))
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

    x = np.load('softmax_x.npy')
    t = np.load('softmax_t.npy')
    x = normalize(x)
    print x.shape, t.shape
    # print x

    N, D = x.shape
    K = 3
    y0 = 1.5 # \xce in paper

    model_mcmc = ff.MulticlassLogisticModel(x, t, K, th0=th0, y0=y0)
    num_iter_mcmc, th_mcmc, neg_log_post_mcmc = run_model(model_mcmc)

    np.savetxt('softmax_num_iter_mcmc-3.csv', np.array(num_iter_mcmc))
    np.savetxt('softmax_th_mcmc-3.csv', np.array(th_mcmc))
    np.savetxt('softmax_neg_log_post_mcmc-3.csv', np.array(neg_log_post_mcmc))

    model_flymc = ff.MulticlassLogisticModel(x, t, K, th0=th0, y0=y0)
    num_iter_flymc, th_flymc, neg_log_post_flymc = run_model(model_flymc, q=0.1, fly=True) # q = prob(dim -> bright)

    np.savetxt('softmax_num_iter_flymc-3.csv', np.array(num_iter_flymc))
    np.savetxt('softmax_th_flymc-3.csv', np.array(th_flymc))
    np.savetxt('softmax_neg_log_post_flymc-3.csv', np.array(neg_log_post_flymc))

    # _model = ff.MulticlassLogisticModel(x, t, K, th0=th0)    # dummy model used to optimize th
    # th = np.random.randn(K, D) * th0
    # def obj_fun(th):
    #     # TODO: th -> th_mat
    #     th_mat = np.reshape(th, [K, D])
    #     return -1.0*_model.log_p_marg(th_mat)
    # def grad_fun(th):
    #     # TODO: th -> th_mat
    #     th_mat = np.reshape(th, [K, D])
    #     grad_mat = -1.0*_model.D_log_p_marg(th_mat)
    #     # TODO: convert to vector
    #     grad = np.reshape(grad_mat, [K*D,])
    #     return grad
    #
    # th_map = optimize.minimize(
    #         # fun=lambda th: -1.0*_model.log_p_marg(th),
    #         fun=obj_fun,
    #         x0=np.random.randn(K * D) * th0,
    #         # jac=lambda th: -1.0*_model.D_log_p_marg(th),
    #         jac=grad_fun,
    #         method='BFGS',
    #         options={
    #             'maxiter': 100,
    #             'disp': True
    #         })
    #
    # TODO: th_map.x is a vector after optimize.minimize, need to convert
    # to matrix for MulticlassLogisticModel
    # th_map = np.reshape(th_map.x, [K, D])
    # np.save('softmax_th_map.npy', th_map)

    th_map = np.load('softmax_th_map.npy')
    model_flymc_map = ff.MulticlassLogisticModel(x, t, K, th0=th0, th_map=th_map)
    num_iter_flymc_map, th_flymc_map, neg_log_post_flymc_map = run_model(model_flymc_map, q=0.01, fly=True)

    # output traces to -3.csv


    np.savetxt('softmax_num_iter_flymc_map-3.csv', np.array(num_iter_flymc_map))
    np.savetxt('softmax_th_flymc_map-3.csv', np.array(th_flymc_map))
    np.savetxt('softmax_neg_log_post_flymc_map-3.csv', np.array(neg_log_post_flymc_map))

if __name__ == "__main__":
    main()
