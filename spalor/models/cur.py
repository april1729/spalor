from scipy.sparse.linalg import svds
from numpy.linalg import pinv
import numpy as np

class CUR():
    def __init__(self,A,n_components):
        self.A=A
        self.n_components=n_components


    def fit(self):
        u,s,vt=svds(self.A,self.n_components)

        ls_u=np.sum(np.square(self.U),0)
        ls_v=np.sum(np.square(self.Vt),1)


        self.cols=[]
        self.C=self.A[:,self.cols]


        self.cols=[];
        self.R=self.A[self.rows,:]


        self.U=pinv(self.C).dot(self.A).dot(pinv(self.R))


    def transform(self):
        pass

    def get_params(self):
        return (self.C, self.U, self.R)