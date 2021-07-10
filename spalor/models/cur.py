from .scipy.sparse.linalg import svds
from numpy.linalg import pinv
import numpy as np

class CUR():
    def __init__(self,A,n_rows=0,n_cols=0, r=0):
        self.A=A
        (d1,d2)=A.shape

        if n_rows:

            self.n_rows=n_rows
        else:
            self.n_rows=d1


        if n_cols:

            self.n_cols=n_cols
        else:
            self.n_cols=d2

        if r:

            self.r=r
        else:
            self.r=min(self.n_cols,self.n_rows)


    def fit(self):
        u,s,vt=svds(self.A,self.r)
        self.U=u;
        self.S=s;
        self.V=vt.transpose();

        ls_u=np.sum(np.square(self.U),0)
        ls_v=np.sum(np.square(self.V),0)


        self.cols=[np.argsort(ls_u)[-self.n_rows:]]
        self.C=np.squeeze(self.A[:,self.cols])

        self.rows=[np.argsort(ls_v)[-self.n_cols:]]
        self.R=np.squeeze(self.A[self.rows,:])
        self.U=pinv(self.C).dot(self.A).dot(pinv(self.R))


    def transform(self):
        pass

    def get_params(self):
        return (self.C, self.U, self.R)