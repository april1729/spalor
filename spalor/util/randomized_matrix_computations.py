import numpy as np
import sklearn.random_projection
from scipy import fftpack


def leverage_scores_QR(X):
    (m,n)=X.shape
    Q,R=np.linalg.qr(X)
    return np.square(np.apply_along_axis(np.linalg.norm, 1,Q))/n


def leverage_scores_aprx(X, c1, c2):
    (m,n)=X.shape
    SX=np.transpose(np.linalg.pinv(fftpack.dct(np.transpose(X),type=2, n=c1)))
    transformer=sklearn.random_projection.GaussianRandomProjection(c2)
    SXPI = transformer.fit_transform(SX)
    return (m/c1)*np.square(np.apply_along_axis(np.linalg.norm, 1, X.dot(SXPI)))/n


