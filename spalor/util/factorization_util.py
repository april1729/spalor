import numpy as np

def partXY(U,V, X):
    '''
    returns a vector of a sparse set of entries of UV^T
    np.sum(np.multiply(U[X[0][:], :],V[X[1][:],:]), axis=1)
    :param U:
    :param V:
    :param X: (2,n) nparray of indices for the entries of UV^T needed
    :return: y (n,) nparray of entries of UV^T
    '''
    return np.sum(np.multiply(U[X[0][:], :],V[X[1][:],:]), axis=1)

def svd_low_rank_plus_sparse(U,Sigma,V, S, eps=1e-6, max_iter=100):
    '''

    Uses power iteration method to find the truncated singular value decompositon of the rank-r approximation to the
    matrix U Sigma V^T +S efficiently

    :param U: (d1,r)
    :param Sigma: (r,r)
    :param V: (d2,r)
    :param S: sparse matrix (d1, d2)
    :return'
    '''




    pass

def svd_from_factorization(U,V):
    '''
    Orthonormalizes U and V to obtain the singular decomposition of UV^T

    :param U: (d1,r) numpy array
    :param V: (d2,r) numpy array
    :return: (U,Sigma,V) - the singular value decomposition of UV^T
    '''

    pass
