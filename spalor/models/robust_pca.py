from spalor.algorithms.rpca_algorithms import *

class RobustPCA():
    def __init__(self, m, n, rank):
        self.m = m
        self.n = n
        self.rank = rank
        self.user_means = np.zeros(m)
        self.user_std = np.zeros(m)

    def fit(self,M):
        (U,Sigma,V,S) = altProjNiave(self.M, self.rank, self.sparsity, fTol=1e-10, maxIter=100)
        self.U=U
        self.Sigma=Sigma
        self.V=V
        self.outliers=S

    def transform(self,X):
        return X.dot(self.V)

    def inverse_transform(self, X):
        return (self.U).dot(X)
