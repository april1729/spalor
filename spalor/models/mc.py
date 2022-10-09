from spalor.algorithms.mc_algorithms import *
from spalor.util.factorization_util import *
import numpy as np

class MC:
    '''
    Matrix completion.  There are two main ways to use this class:

        - PCA when some proportion of the data is missing.  This class will calculate the principal components with the data available.  This can be used to fill in the missing data, or the principal components and scores can be used on their own as if the data was never missing to begin with.
        - A supervised machine learning algorithm based on collaborative filtering.  Typically, this is thought as a recommendation system where d1 is the number of users, d2 is the number of items, and the values are the users ratings on the items. The features are the index of the user and the item, and the target variable is the rating.

    See the `user guide <http://www.spalor.org/user_guide/matrix_completion>` for a detailed description

    Parameters
    ----------
    n_components : int, default = 10
        Number of principle components to solve for, that is, the rank of the matrix to be completed. If set to a number between 0 ad 1, the parameter will be taken to be the ratio of the smallest singular value to the largest.

    solver : {'lmafit', 'svt', 'alt_min', 'alt_proj'}, default='lmafit'
        solver to use  see ../algorithms/mc_algorithms

    lambda : float, must be larger than 0, default 0.5
        Regularization parameter.  Only used if solver='svt' or 'apgd'.

        Increasing the parameter reduces overfiting, but may lead to estimaiton bias towards zero, particularly with solver='svt'

    tol : float, default=1e-6
        Stopping  criteria for matrix completion solver.

    Attributes
    -----------
    d1 : int
        Number of rows in matrix (typically, the number of samples in the dataset)

    d2 : int
        Number of columns in the matrix (typically, the number of features in the dataset)

    U : ndarray of size (d1, n_components)
        left singular vectors

    S : ndarray of size (n_components,)
        singular values

    V : ndarray of size (d2, n_components)
        right singular vectors.

    T : ndarray of size (d1, n_components)
        Score matrix, U*S.  Often used for classification from PCA.

    components : ndarray of size (d2, n_components)
        Principal axes in feature space, representing the directions of maximum variance in the data.
    '''
    def __init__(self, n_components=10):
        self.rank = n_components

    def fit(self, X, y = None, missing_val="nan", normalize=False, method="lmafit"):
        '''

        Parameters
        ----------
        X :

        y :

        '''

        if y is None:
            (self.d1,self.d2)=X.shape
            if missing_val=="nan":
                [I,J]=np.where(~ np.isnan(X))
            else:
                [I,J]=np.where(~ (X == missing_val))
            X_fit=np.vstack([I,J]).T
            y=X[I,J]

        else:
            if X_fit.shape[0]==2 and X_fit.shape[1]>2:
                X_fit=X_fit.T

            self.d1=max(X[:,0])+1;
            self.d2=max(X[:,1])+1;
            X_fit=X

        if normalize:
            self.user_means = np.zeros(self.d1)
            self.user_std = np.zeros(self.d1)

            y_fit = np.zeros(len(y))
            for user in range(0, self.m):
                idx = np.where(X[:,0] == user)
                self.user_means[user] = np.mean(y[idx])
                self.user_std[user] = np.std(y[idx])
                y_fit[idx] = (y[idx] - self.user_means[user]) / self.user_std[user]
        else:
            y_fit=y;

        if method=="lmafit":
            (U, V) = lmafit(self.d1, self.d2, self.rank, X_fit.T, y_fit)

            (u,s,v)= svd_from_factorization(U,V)
            self.U = u.dot(np.diag(s))
            self.V = v


    def fit_transform(self, X, y=None):
        '''
        Parameters
        ----------
        X :

        y :

        '''
        self.fit(X,y=y)
        return self.U

    def transform(self,X):
        return X.dot(self.V.T)

    def inverse_transform(self,X):
        return X.dot(self.V)

    def predict(self, X):
        '''
        Parameters
        ----------
        X

        Returns
        -------

        '''
        y = partXY(self.U, self.V, X)

        for user in range(0, self.m):
            idx = np.where(X[0, :] == user)
            y[idx] = (y[idx] * self.user_std[user]) + self.user_means[user]
        return y


    def get_covariance(self):
        """
        Calculates an estimate of covariance matrix.

        Entry (i,j) will be a the correlation between feature i and feature j.  A value close to 1 is a strong postive correlatio, a value close to -1 is a strong negative correlation, and a value close to 0 is no correlation.

        Returns
        -------
        cov : array, shape=(d2, d2)
            Estimated covariance of data.
        """

        return self.V.dot(self.S**2).dot(self.V.transpose())/(d1-1)


    def to_matrix(self):
        '''
        Calculates the completed matrix.

        Warning: In some cases, this may be to large for memory.  For example, when being used for recommendation systems.

        Returns
        -----------
        M : ndarray of size (d1,d2)
            Completed matrix
        '''

        return self.V.dot(self.S).dot(self.U)
