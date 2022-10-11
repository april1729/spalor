from math import *

from scipy.sparse import coo_matrix

from spalor.regularization.thresholding import *
from spalor.matrix_tools.factorization_util import *
import numpy as np
from scipy.optimize import minimize

def lmafit(d1, d2, r, known, data):
    '''
    A rank-constrained matrix completion algorithm that uses a successive over-relaxation scheme.  

    Links for more details:
    - http://lmafit.blogs.rice.edu/

    Parameters
    ----------
    d1 : int
        number or rows in matrix
    d2 : int
        number of columns in matrix
    r : int 
        target rank of matrix.
    known : np array
        array with 2 columns and many rows, indicating indices of the matrix you have a measurement of
    data : np array
        vector of measurements, in same order as 'known'  

    References
    ----------
    .. [1] Wen, Z., Yin, W. & Zhang, Y. Solving a low-rank factorization model for matrix completion by a nonlinear successive over-relaxation algorithm. Math. Prog. Comp. 4, 333–361 (2012). https://doi.org/10.1007/s12532-012-0044-1


    '''   

    # set parameters
    # TODO: parameter selection
    tol = 1e-5;
    maxit = 1000;
    iprint = 2;
    est_rank = 1;
    rank_max = max(floor(0.1 * min(d1,d2)), 2 * r);
    rank_min = 1;
    rk_jump = 10;
    init = 0;
    save_res = 0;

    # Initialize Variables

    X = np.random.randn(d1, r)
    Y = np.random.randn(d2, r)
    Res = data - partXY(X, Y, known)
    S = coo_matrix((Res, known), shape=(d1,d2))
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

    '''
    A very simple matrix completion algorithm comprising of two steps at each iteration:
        - project L onto set of matrices satisfying the measurements
        - project L onto the set of rank r matrices

    Parameters
    ----------
    m : int
        number or rows in matrix
    n : int
        number of columns in matrix
    r : int 
        target rank of matrix.
    X : np array
        array with 2 columns and many rows, indicating indices of the matrix you have a measurement of
    y : np array
        vector of measurements, in same order as 'known'  

    References
    ----------
    .. [2] Tanner, J., & Wei, K. (2013). Normalized iterative hard thresholding for matrix completion. SIAM Journal on Scientific Computing, 35(5), S104-S125.
    '''   


    L=np.zeros((m,n))


    for iter in range(0,2000):
        L[X[0,:],X[1,:]]=y;
        L=lowRankProj(L,r+max(0,round(10-iter/100)));
        # print(np.linalg.norm(L[X[0,:],X[1,:]]-y))
    u,s,vt=svds(L,r)
    U=u.dot(np.sqrt(s));
    V=np.sqrt(s).dot(vt);
    
    return (U,V)
    
def svt(m, n, beta_max, known, data, eps=1e-5, r_max=None):
    '''
    Singular value thresholding for matrix completion 

    Parameters
    ----------
    m : int
        number or rows in matrix
    n : int
        number of columns in matrix
    beta_max : float 
        largest singular value to keep.  Larger values mean less regularization, and less estimtor bias
    known : np array
        array with 2 columns and many rows, indicating indices of the matrix you have a measurement of
    data : np array
        vector of measurements, in same order as 'known'  
    eps : float
        stopping criteria.  (default: 1e-5)
    r_max : int
        upper bound on the rank of the matrix.  The smaller this is, the faster the algorithm will be

    References
    ----------
    .. [3] Cai, J. F., Candès, E. J., & Shen, Z. (2010). A singular value thresholding algorithm for matrix completion. SIAM Journal on optimization, 20(4), 1956-1982.
    '''   

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
    '''
    A very simple algorithm for matrix completion via alternating minimization.

    Parameters
    ----------
    m : int
        number or rows in matrix
    n : int
        number of columns in matrix
    Omega : np array
        array with 2 columns and many rows, indicating indices of the matrix you have a measurement of
    known : np array
        vector of measurements, in same order as 'known'  
    '''   

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
