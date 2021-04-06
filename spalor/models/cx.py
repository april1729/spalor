from scipy.sparse.linalg import svds
from numpy.linalg import pinv
import numpy as np

from spalor.util.randomized_matrix_computations import *

class CX():
    def __init__(self,n_components=10):
        self.n_components=n_components

    def fit(self,A):
        self.A=A
        ls=np.power(leverage_scores_QR(np.transpose(A),r=3),2)
        ls=ls/np.sum(ls)
        np.transpose(A)
        #self.cols=[np.argsort(ls)[-self.n_components:]]
        self.cols=np.random.choice(len(ls), self.n_components, p=ls)

        self.C=np.squeeze(self.A[:,self.cols])
        self.X=pinv(self.C).dot(self.A)


    def fit_from_SVD(self, U, S, V):
        ls=np.sum(np.square(self.Vt),1)
        self.cols=[np.argsort(ls)[-self.n_components:]]
        self.C=self.A[:,self.cols]
        self.X=pinv


    def transform(self,A):
        return pinv(self.C).dot(self.A)

    def fit_transform(self,A):
        self.fit(A)
        return self.X

    def get_params(self):
        return (self.C, self.X,self.cols)
