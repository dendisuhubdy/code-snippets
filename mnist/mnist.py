'''
Tutorial Deep Learning
'''


import cPickle
import gzip
import numpy

def load_mnist():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    return train_set, valid_set, test_set


