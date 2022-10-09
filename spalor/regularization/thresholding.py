import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import svds


def sparseProj(x, s):
    '''
    Given an np array x of any size, returns the projection onto the set ||x||_0 <=s.
    :param x: np array of any size
    :param s: number of nonzero entries we want
    :return: nparray of the same size as x with ||x||_0 <=s
    '''
    x_out=x;
    threshold=np.sort(abs(x_out.flatten()))[-s]
    return sparseHardThresholding(x_out, threshold)

def sparseHardThresholding(x, t):
    xThresh = x
    xThresh[abs(xThresh) < t] = 0
    return xThresh

def sparseSoftThresholding(x,t):
    xThresh=np.sign(x)*np.maximum(0, abs(x)-t)
    return xThresh


def lowRankProj(x,k):
    u,s,vt = svds(x,k)
    return u.dot(np.diag(s)).dot(vt)

def lowRankProxThresholding(X,proxFunc,r=None):
    if r is not None:
        u,s,vt = svds(x,r)
    else:
        u,s,vt = svd(x, full_matrices=False)

    s=proxFunc(s)
    return u.dot(np.diag(s)).dot(vt)

def lowRankSoftThresholding(x,t,r=None):
    if r is not None:
        u,s,vt = svds(x,r)
    else:
        u,s,vt = svd(x, full_matrices=False)

    s=sparseSoftThresholding(s,t)
    return u.dot(np.diag(s)).dot(vt)

def singularValue(M, i):
    (u,s,v)=svds(M,i+1)
    return min(s)

if __name__ == "__main__":
    x_test=np.array([[1,2,3],[4,5,6]])

    x_hard_35=np.array([[0,0,0],[4,5,6]])
    x_soft_35=np.array([[0,0,0,],[0.5,1.5,2.5]])
    x_proj_2=np.array([[0,0,0],[0,5,6]])
    print("sparseHardThresholding error:", np.linalg.norm(x_hard_35-sparseHardThresholding(x_test, 3.5)))
    x_test=np.array([[1,2,3],[4,5,6]])
    print("sparseProj error:", np.linalg.norm(x_proj_2-sparseProj(x_test, 2)))
    x_test=np.array([[1,2,3],[4,5,6]])
    print("sparseSoftThresholding error:", np.linalg.norm(x_soft_35-sparseSoftThresholding(x_test, 3.5)))


