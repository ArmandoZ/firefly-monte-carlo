import numpy as np
import os
import sys
from mnist import MNIST
from sklearn.decomposition import PCA

def preprocess_mnist_data():
    curDir = os.path.dirname(os.path.realpath(__file__))
    mndata = MNIST(curDir)
    mndata.load_training()
    ss_idx = filter(lambda i: mndata.train_labels[i] in [7,9], range(len(mndata.train_images)))
    data_ss = np.array(mndata.train_images)[ss_idx,:]
    label_ss = np.array(mndata.train_labels)[ss_idx]
    pca = PCA(n_components=50)
    pca.fit(data_ss)
    x = pca.transform(data_ss)
    x = np.concatenate((x, np.ones((x.shape[0],1))),axis=1)
    t = label_ss == 7

    np.save('{0}/logistic_x.npy'.format(curDir), x)
    np.save('{0}/logistic_t.npy'.format(curDir), t)

def load_logistic_data():
    '''
    Read the data for logistic regression experiment.
    7s and 9s images in MNIST dataset, only use 50 principal components.
    Cached in logistic_x.npy logistic_t.npy
    '''
    curDir = os.path.dirname(os.path.realpath(__file__))
    if not (os.path.exists('{0}/logistic_x.npy'.format(curDir)) and os.path.exists('{0}/logistic_t.npy'.format(curDir))):
        preprocess_mnist_data()
    x = np.load('{0}/logistic_x.npy'.format(curDir))
    t = np.load('{0}/logistic_t.npy'.format(curDir))
    return x, t

def main():
    # x, t = get_softmax_data()
    # print x.shape, t.shape
    pass

if __name__ == '__main__':
    main()
