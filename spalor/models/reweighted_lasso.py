import numpy as np
from .spalor.algorithms.lasso_algorithms import *
class ReweightedLasso():
# TODO: Add a cross validation routine to select the best value of alpha


    def __init__(self):
        pass

    def fit(self, X,y, alpha=1):
        self.X=X
        self.y=y
        self.alpha=alpha
        (self.n,self.d)=X.shape

        self.XX=np.transpose(X).dot(X)
        self.Xy=np.transpose(self.X).dot(self.y)

        self.w=sparse_prox_grad(self.XX, self.Xy, self.alpha)


    def predict(self,X):
        return X.dot(self.w)


    def get_params(self):
        return self.w

    def coef_(self):
        return self.w