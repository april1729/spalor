import numpy as np

def partXY(x,y, known):
    return np.sum(np.multiply(x[known[0][:], :],y[known[1][:],:]), axis=1)


def mat2tensor(X,d1,d2):
    # TODO mat2tensor
    return X

def tensor2mat(X):
    # TODO tensor2mat
    return X

def vec(X):

    return X