import numpy as np
import sklearn.random_projection
from scipy import fftpack
from scipy.sparse.linalg import svds

def leverage_scores_QR(X,r=10):
    ''''(m,n)=X.shape
    Q,R=np.linalg.qr(X)
    ls= np.square(np.apply_along_axis(np.linalg.norm, 1,Q))/n'''

    (m, n) = X.shape
    u,s,v=svds(X,r)
    ls=np.square(np.apply_along_axis(np.linalg.norm, 1,u))/n

    #Q,R=np.linalg.qr(X)
    #ls= np.square(np.apply_along_axis(np.linalg.norm, 1,Q))/n

    return ls


def leverage_scores_aprx(X, c1, c2):
    (m,n)=X.shape
    SX=np.transpose(np.linalg.pinv(fftpack.dct(np.transpose(X),type=2, n=c1)))
    transformer=sklearn.random_projection.GaussianRandomProjection(c2)
    SXPI = transformer.fit_transform(SX)
    ls=np.square(np.apply_along_axis(np.linalg.norm, 1, X.dot(SXPI)))
    ls=m*ls/np.sum(ls)
    return ls
