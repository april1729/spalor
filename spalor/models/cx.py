from scipy.sparse.linalg import svds
from numpy.linalg import pinv
import numpy as np

from spalor.util.randomized_matrix_computations import *

class CX():
    def __init__(self,A,n_components):
        self.n_components=n_components

    def fit(self,A):
        self.A=A
        ls=leverage_scores_QR(A)
        self.cols=[np.argsort(ls)[-self.n_components:]]
        self.C=self.A[:,self.cols]
        self.X=pinv(self.C).dot(self.A)


    def fit_from_SVD(self, U, S, V):
        ls=np.sum(np.square(self.Vt),1)
        self.cols=[np.argsort(ls)[-self.n_components:]]
        self.C=self.A[:,self.cols]
        self.X=pinv(self.C).dot(self.A)


    def transform(self,A):
        return pinv(self.C).dot(self.A)

    def fit_transform(self,A):
        self.fit(A)
        return self.X

    def get_params(self):
        return (self.C, self.X,self.cols)