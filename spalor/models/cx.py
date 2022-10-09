from scipy.sparse.linalg import svds
from numpy.linalg import pinv
import numpy as np

from spalor.util.randomized_matrix_computations import *


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

    solver : {'leverage_scores_QR', 'leverage_scores_approx', 'group_sparse_regression'}, default='leverage_scores_QR'
        choice of three different solvers.  

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
    '''

    def __init__(self, n_components=10, solver='leverage_scores_QR'):
        self.n_components = n_components
        self.solver=solver

    def fit(self, A):
        self.A = A

        ls = np.power(leverage_scores_QR(np.transpose(A), r=3), 2)
        ls = ls / np.sum(ls)
        np.transpose(A)

        self.cols = np.random.choice(len(ls), self.n_components, p=ls)

        self.C = np.squeeze(self.A[:, self.cols])
        self.X = pinv(self.C).dot(self.A)

    def fit_from_SVD(self, U, S, V):
        '''
        Used to fit the model when the singular value decomposition is already known.  Knowing the SVD a priori greatly reduces the computional time of the method.

        Parameters:
        -------------
        U : ndarray
        S : ndarray
        V : ndarray  
        '''
        ls = np.sum(np.square(self.Vt), 1)
        self.cols = np.random.choice(len(ls), self.n_components, p=ls)
        self.C = self.A[:, self.cols]
        self.X = pinv(self.C).dot(self.A)


    def transform(self, C):
        return C.dot(self.X)

    def fit_transform(self, A):
        self.fit(A)
        return self.C

    def inverse_transform(self, X):
        return self.C.dot(X)

    def get_params(self):
        return (self.C, self.X, self.cols)


    def get_covariance(self):
        return self.X.T.dot(self.X)/n_components
