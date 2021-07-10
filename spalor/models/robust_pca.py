from ..algorithms.rpca_algorithms import *
from scipy.sparse.linalg import svds
class RPCA():
    def __init__(self, r=10, sparsity=0.05):
        self.r = r
        self.sparsity = sparsity

    def fit(self,M):
        self.M=M;
        d1,d2=M.shape
        if self.sparsity<1:
            self.sparsity=round(d1*d2*self.sparsity)
        (L,S) = altProjNiave(self.M, self.r, self.sparsity, fTol=1e-10, maxIter=1000)
        #(L,S) = altProj(self.M,r=self.r)
        u,s,vt=svds(L,self.r)
        self.U=np.reshape(u.dot(np.diag(s)),(d1,self.r))
        self.V=vt.transpose()
        self.S=S

    def transform(self,X):
        return X.dot(self.V)

    def inverse_transform(self, X):
        return (self.U).dot(X)
    def get_params(self):
        return(self.U, self.V, self.S)
