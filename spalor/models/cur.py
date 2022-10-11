from scipy.sparse.linalg import svds
from numpy.linalg import pinv
import numpy as np
from spalor.matrix_tools import leverage_score
class CUR:
    '''
    Dimensionality reduction based on a low-rank matrix faactorization:
        A=C*U*R
    where C consists of columns sampled from A, R is rows sampled from A, and U=(C'*C)^-1 *C' *A  * R'*(R*R')^-1.

    Typically, the columns and rows are selected at random with probabilites proportional to the leverage scores.


    Parameters:
    ------------
    n_rows : int, default=10
        Number of rows to sample.
    n_cols : int, default=10
        Number of columns to sample.

    method : {'exact', 'approximate', 'random'}, default='leverage_scores_QR'
        method to selct rows.
            - "exact": randomly select by leverage scores
            - "approximate" : randomly select columns by approximated leverage scores
            - "random" : randomly select columns

    Attributes:
    ------------
    d1 : int
        number or rows in the original matrix
    d2 : int
        number of columns in the original matrix
    cols : list 
        list containing indices of columns sampled
    C : ndarray, shape = (d1,n_components)
        Columns sampled
    X : ndarray, shape = (n_components, d2)
        Score matrix, often used for classification. Coordinates in the lower dimensional column space

    Example:
    --------
    ```
    A=np.array([[1, 1, 2, 2],
        [2, 1, 3, 5],
        [1, 2, 3, 1],
        [3, 1, 4, 8]], dtype=float)

    cur=CUR(n_rows=2, n_cols=2)
    cur.fit(A)

    print("C:\n", cur.C)
    print("U:\n", cur.U)
    print("R:\n", cur.R)
    print("columns used: \n", cur.cols)
    print("rows used: \n", cur.rows)
    ```
    '''

    def __init__(self,n_rows=10,n_cols=10, r=None):
        self.n_rows=n_rows
        self.n_cols=n_cols
        self.r=r

        if self.r is None:
            self.r=max(self.n_cols,self.n_rows)

    def fit(self, A, rows=None, cols=None):
        '''
        fit matrix A to CUR model
        '''

        self.A=A
        (d1,d2)=A.shape

        svdA=svds(self.A,self.r)

        self.svdA=svdA

        if cols==None:
            ls_u=leverage_score(svdA, k=self.n_cols, axis=1)
            cols = np.random.choice(len(ls_u), self.n_cols, p=ls_u)
        if rows ==None:
            ls_v=leverage_score(svdA, k=self.n_rows, axis=0)
            rows = np.random.choice(len(ls_v), self.n_rows, p=ls_v)


        self.cols=[np.argsort(ls_u)[-self.n_rows:]]
        self.rows=[np.argsort(ls_v)[-self.n_cols:]]

        self.C=np.squeeze(self.A[:,self.cols])
        self.R=np.squeeze(self.A[self.rows,:])

        self.U=pinv(self.C).dot(self.A).dot(pinv(self.R))


    def transform(self,X):
        return np.squeeze(X[:,self.cols])

    def fit_transforms(self,X):
        self.fit(X)
        return self.C

    # def inverse_transform(X):
    #     return self.X.dot(self.U).dot(self.R)

    def get_params(self):
        return (self.C, self.U, self.R)


if __name__=="__main__":
    A=np.array([[1, 1, 2, 2],
            [2, 1, 3, 5],
            [1, 2, 3, 1],
            [3, 1, 4, 8]], dtype=float)

    cur=CUR(n_rows=2, n_cols=2)
    cur.fit(A)

    print("C:\n", cur.C)
    print("U:\n", cur.U)
    print("R:\n", cur.R)
    print("columns used: \n", cur.cols)
    print("rows used: \n", cur.rows)