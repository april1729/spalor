from math import *

from scipy.sparse import coo_matrix

from spalor.regularization.thresholding import *
from spalor.util.factorization_util import *
import numpy as np
from scipy.optimize import minimize

def lmafit(m, n, r, known, data):
    L = len(data);

    # set parameters
    # TODO: parameter selection
    tol = 1e-5;
    maxit = 150;
    iprint = 2;
    est_rank = 1;
    rank_max = max(floor(0.1 * min(m, n)), 2 * r);
    rank_min = 1;
    rk_jump = 10;
    init = 0;
    save_res = 0;

    # Initialize Variables

    X = np.random.rand(m, r)
    Y = np.random.rand(n, r)
    Res = data - partXY(X, Y, known)
    S = coo_matrix((Res, known), shape=(m, n))
    # S=np.zeros((m,n))
    alf = 0
    increment = 0.1

    Res = data - partXY(X, Y, known)

    res = np.linalg.norm(Res);

    # main loop

    for iter in range(0, maxit):
        X0 = X
        Y0 = Y
        Res0 = Res
        res0 = res
        print(res)
        
        X = X + S.dot(Y).dot(np.linalg.pinv(Y.transpose().dot(Y)))
        XXInv = np.linalg.pinv(X.transpose().dot(X));
        Y = Y.dot(X0.transpose().dot(X)).dot(XXInv) + S.transpose().dot(X).dot(XXInv)
        Res = data - partXY(X, Y, known)

        res = np.linalg.norm(Res);
        ratio = res / res0;
        '''
        print("iter: ",iter," residual: ",res,"alf", alf)
        if ratio >= 1:
            increment = max(0.1 * alf, 0.1 * increment)
            X = X0
            Y = Y0
            Res = Res0
            res = res0
            alf = 0
        elif ratio > 0.7:
            increment = max(increment, 0.25 * alf)
            alf = alf + increment
        '''
        S = coo_matrix(((alf + 1) * Res, known))

    return (X, Y)
def alt_proj(m,n,r,X,y):
    L=np.zeros((m,n))


    for iter in range(0,2000):
        L[X[0,:],X[1,:]]=y;
        L=lowRankProj(L,r+max(0,round(10-iter/100)));
        print(np.linalg.norm(L[X[0,:],X[1,:]]-y))
    u,s,vt=svds(L,r)
    U=u.dot(np.sqrt(s));
    V=np.sqrt(s).dot(vt);
    
    return (U,V)
def svt(m, n, beta_max, known, data, eps=1e-5, r_max=None):
    if r_max is None:
        r_max = min(m, n)
    beta = beta_max / (1.2 ** 30)
    maxIter = 100;
    X = np.zeros((m, n))

    for iter in range(0, maxIter):
        X[known] = data
        X = lowRankSoftThresholding(X, 1 / beta, r_max)
        beta = min(beta_max, beta * 1.2)
    return X

def alt_min(m,n,r, Omega, known):
    U=np.random.rand(m,r)
    V=np.random.rand(r,n)

    for i in range(0,100):   
        
        objU=lambda x: np.linalg.norm(np.reshape(x, [m,r]).dot(V)[Omega]-known)**2
        U = np.reshape(minimize(objU, U).x, [m,r])
        
        objV=lambda x: np.linalg.norm(U.dot(np.reshape(x, [r,n]))[Omega]-known)**2
        V = np.reshape(minimize(objV, V).x, [r,n])

        res=np.linalg.norm(U.dot(V)[Omega]-known)
        if res < 0.0001:
            break
    return (U,V)

def partXY(x, y, known):
    return np.sum(np.multiply(x[known[0][:], :], y[known[1][:], :]), axis=1)


if __name__ == "__main__":
    known = [[0, 0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 3, 0, 1, 2, 0, 1, 3]]
    y = [1, 1, 2, 2, 2, 1, 3, 1, 2, 1]
    m = 3
    n = 4
    r = 2

    # (U, V) = lmafit_mc_adp(3, 4, 2, known, y)
    # print(U.dot(V.transpose()))

    X = svt(3, 4, 10, known, y, r_max=2)

    print(X)
