import numpy as np
from ..models.reweighted_lasso import *
from ..algorithms.lasso_algorithms import *


def test_lasso():
    n=10000
    d=1000
    r=30

    X=np.random.randn(n,d)
    w=np.zeros((d,))
    w[0:r]=np.random.randn(r)
    w[0:r]= w[0:r]+np.sign(w[0:r])
    y=X.dot(w)+np.random.randn(n)

    XX=np.transpose(X).dot(X)
    Xy=np.transpose(X).dot(y)

    w_lasso=sparse_prox_grad(XX, Xy,3)

    print(w_lasso)


if __name__ is "__main__":
    test_lasso()