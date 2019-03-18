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

def main():
    # x, t = get_softmax_data()
    # print x.shape, t.shape

if __name__ == '__main__':
    main()
