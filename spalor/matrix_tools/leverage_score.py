import numpy as np
import sklearn.random_projection
from scipy import fftpack
from scipy.sparse.linalg import svds



def leverage_score_exact(X,r=10):

    (m, n) = X.shape
    u,s,v=svds(X,r)
    ls=np.square(np.apply_along_axis(np.linalg.norm, 1,u))

    #Q,R=np.linalg.qr(X)
    #ls= np.square(np.apply_along_axis(np.linalg.norm, 1,Q))/n

    return ls


def leverage_score_aprx(X, c1, c2):
    (m,n)=X.shape
    SX=np.transpose(np.linalg.pinv(fftpack.dct(np.transpose(X),type=2, n=c1)))
    transformer=sklearn.random_projection.GaussianRandomProjection(c2)
    SXPI = transformer.fit_transform(SX)
    ls=np.square(np.apply_along_axis(np.linalg.norm, 1, X.dot(SXPI)))
    ls=m*ls/np.sum(ls)
    return ls

def leverage_score(A, k=10, method="exact", axis=0):
    '''
    Calcluates the leverage statistic for each row (or column) of X when calculating the rank k approximation of A.

    Parameters:
    -------------
    A: either a n by d np-array or a tuple containing the SVD of A (the output of np.linalg.svd or the output of scipy.sparse.linalg.svds)
    k: rank for which the leverage statitics of the low rank approximation are calculated for
    method: If exact, calculate leverage scores using the rank-k svd [1].  if approximate, use the Fast Johnson-Lindenstrauss Transform to approximate leverage scores[2].
    axis: dimension of the matrix to calclute leverage scores for (0: calculate score for columns, 1: calculate for rows)
    
    Returns:
    -----------
    l: vector of leverage scores with length A.shape[axis]

    References:
    ------------
    [1] Randomized algorithms for matrices and data, Michael W. Mahoney (page 7), https://arxiv.org/pdf/1104.5557.pdf

    [2] Fast Approximation of Matrix Coherence and Statistical Leverage, Petros Drineas and Malik Magdon-Ismail and Michael W. Mahoney and David P. Woodruff, JMLR, https://www.stat.berkeley.edu/~mmahoney/pubs/coherence-jmlr12.pdf
    '''



    if type(A) ==tuple and len(A)==3: # if svd is provided
        svdA=A

        u=svdA[0]
        s=svdA[1]
        v=svdA[2]

        if axis==1:
            (u, v)= (v.T, u.T)

        n=u.shape[0]
        d=v.shape[1]

        if len(s)>k: 
            '''
            if more singular values than needed are calculated, get the top k. 
            Note that different SVD solvers give results in different orders, 
            so dont make assumptions about the order
            '''

            ind=s.argsort()[-k:][::-1]
            s=s[ind]
            u=u[:, ind]
            v=v[ind, :]

    else: # if svd is not provided
        if axis== 1:
            A=A.T
        (n,d) = A.shape

        if method=='exact':
            [u,s,v]=svds(A,k)


    if method=="exact":
        ls=np.square(np.apply_along_axis(np.linalg.norm, 1,u))/k
    if method=="approximate":
        ls=leverage_score_aprx(A,k, k)/n
    return ls



if __name__=="__main__":
    A=np.array([[1, 1, 2, 2],
                [2, 1, 3, 5],
                [1, 2, 3, 1],
                [3, 1, 4, 8]], dtype=float)
    print(leverage_score(A, k=2, axis=1))
    #%%
    svdA=np.linalg.svd(A)
    print(leverage_score(svdA, k=2, axis=1))
    #%%

    print(leverage_score(A, k=2, axis=1, method="approximate"))



    print(leverage_score(A, k=2, axis=0))
    #%%
    svdA=np.linalg.svd(A)
    print(leverage_score(svdA, k=2, axis=0))
    #%%

    print(leverage_score(A, k=2, axis=0, method="approximate"))

