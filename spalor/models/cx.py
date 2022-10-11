from scipy.sparse.linalg import svds
from numpy.linalg import pinv
import numpy as np
from spalor.matrix_tools import leverage_score


class CX():
    '''
    Dimensionality reduction based on a low-rank matrix faactorization:
        A=C*X
    where C consists of columns sampled from A, and X=(C'*C)^-1 *C' *A.

    Typically, the columns sampled to get C are selected at random with probabilites proportional to the leverage scores.


    Parameters:
    ------------
    n_components : int, default=10
        Number of columns to sample.

    method : {'exact', 'approximate', 'random'}, default='exact'
        method to select rows.
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
    ---------
    ```
    A=np.array([[1, 1, 2, 2],
        [2, 1, 3, 5],
        [1, 2, 3, 1],
        [3, 1, 4, 8]], dtype=float)
    cx=CX(n_components=2)
    X=cx.fit_transform(A)
    print("C:\n", cx.C)
    print("X:\n", cx.X)
    print("columns used: \n", cx.cols)
    ```
    '''

    def __init__(self, n_components=10, method='approximate'):
        self.n_components = n_components
        self.method=method

    def fit(self, A, cols=None, svdA=None):
        '''
        Fit CX model

        Parameters:
        -----------
        A: numpy array with shape (n,d)
            Matrix to fit model to
        cols : (optional) list or 1d numpy array
            list of columns to use.  If specified, `method` and `n_components` are ignored
        svdA : (optional) length 3 tuple 
            the output of `np.linalg.svd` or `scipy.sparse.linalg.svds`.  If you already have the svd of A, specifying it saves on computation.

        Returns:
        ---------
        updated model
        '''

        self.A = A

        n=A.shape[1]

        if cols is None:

            if svdA is not None:
                ls_input=svdA
            else:
                ls_input=A

            ls=leverage_score(ls_input, k=self.n_components, axis=1, method=self.method) **2
            ls=ls/ls.sum()
            cols = np.random.choice(len(ls), self.n_components, p=ls)

        self.cols=cols
        self.C = np.squeeze(self.A[:, self.cols])
        self.Cpinv=pinv(self.C)
        self.X = self.Cpinv.dot(self.A)
        return self

    def transform(self, A):
        """
        Extract columns of A

        Parameters:
        -----------
        A: numpy array with shape (n,d)

        Returns:
        ---------
        Columns of A corresponding to the ones use in the CX model
        """

        return np.squeeze(A[:, self.cols])

    def fit_transform(self, A, cols=None, svdA=None):
        '''
        Fit and return columns
        '''
        self.fit(A, cols=cols, svdA=svdA)
        return self.C

    def inverse_transform(self, C):
        """
        Infer entire matrix from subset of columns

        Params:
        ------
        C: numpy array with shape(n, n_components)

        Returns:
        -------
        ndarray with shape (n,d)
        """

        return C.dot(self.X)

    def get_covariance(self):
        return self.X.T.dot(self.X)/self.n_components

if __name__=="__main__":
    A=np.array([[1, 1, 2, 2],
            [2, 1, 3, 5],
            [1, 2, 3, 1],
            [3, 1, 4, 8]], dtype=float)
    cx=CX(n_components=2)
    X=cx.fit_transform(A)
    print("C:\n", cx.C)
    print("X:\n", cx.X)
    print("columns used: \n", cx.cols)


