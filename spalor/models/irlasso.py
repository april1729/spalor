import numpy as np
from spalor.algorithms.lasso_algorithms import *
class ReweightedLASSO():

    def __init__(self,X, y):
        self.X=X
        self.y=y
        (self.n,self.d)=X.shape

    def fit(self):
        self.cov=np.cov(self.X)
        self.Xy=np.transpose(self.X).dot(self.y)
        alpha=1
        for iter in range(0,100):
            w=prox(w-(alpha/self.n)*(self.cov.dot(w)+self.Xy))
        self.w=w
    def predict(self,X):
        return X.dot(self.w)


    def get_params(self):
        return self.w