import numpy as np

def unpickle(file):
    '''
    unpicke function to read the file into dictnary
    '''
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def get_softmax_data():
    '''
    read The CIFAR-10 dataset and fetch the first three classes
    used for the softmax classification task
    '''
    filename = '../data/data_batch_1'
    data_dict = unpickle(filename)
    x = data_dict['data']               # data
    t = np.array(data_dict['labels'])   # label
    # get the indices of label 0, 1 and 2
    idx_1, idx_2, idx_3 = t == 0, t == 1, t == 2
    idx_1_to_3 = idx_1 + idx_2 + idx_3
    # featch data
    x, t = x[idx_1_to_3], t[idx_1_to_3]
    return x, t

def get_regression_data():
    '''
    read The Harvard Organic Photovoltaics 2015 dataset
    used for the regression experiment
    '''
    filename = '../data/HOPV_15_revised_2.data'
    f = open(filename)
    n = 1
    for line in f:
        print line
        n+=1
        if n == 200:
            exit(1)

def main():
    # x, t = get_softmax_data()
    # print x.shape, t.shape
    get_regression_data()

if __name__ == '__main__':
    main()
