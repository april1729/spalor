from ..algorithms.rpca_algorithms import *
from scipy.sparse.linalg import svds
class RPCA():
    '''
    Robust Principal Component Analysis.  

    Simultaniously performs PCA while identifying and correcting outliers.

    See the `user guide <http://www.spalor.org/user_guide/rpca>` for a detailed description

    Parameters
    ----------
    n_components : int
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
        right singular vectors. Often, these are the prinicipal component axes, or the basis

    T : ndarray of size (d1, n_components)
        Score matrix, U*S.  Often used for classification from PCA.

    outliers : ndarray of size (d1,d2)


    components : ndarray of size (d2, n_components)
        Principal axes in feature space, representing the directions of maximum variance in the data.


    '''
    def __init__(self, r=10, sparsity=0.05):
        self.r = r
        self.sparsity = sparsity

    def fit(self,M):
        '''
        Parameters
        ----------
        M : ndarray 
            observed data matrix with an unknown but sparse set of outliers
        '''

        self.M=M;
        d1,d2=M.shape
        if self.sparsity<1:
            self.sparsity=round(d1*d2*self.sparsity)
        (L,outliers) = altProjNiave(self.M, self.r, self.sparsity, fTol=1e-10, maxIter=1000)
        #(L,S) = altProj(self.M,r=self.r)
        u,s,vt=svds(L,self.r)
        self.U=np.reshape(u.dot(np.diag(s)),(d1,self.r))
        self.V=vt.transpose()
        self.S=S

    def fit_transform(self,M):
        '''
        Parameters
        ----------
        M : ndarray of size (d1,d2)
            observed data matrix with an unknown but sparse set of outliers

        Returns
        ---------
        T : ndarray of size (d1, r)

        '''

        self.fit(M)

        return self.U.dot(self.S)

    def transform(X):
        
        '''
        V is already solved for, so we just need to solve:

        min U, outliers   ||U*V+outliers -X ||_F^2  s.t. outliers is spart
        '''
        

        return (U, outliers)


    def inverse_transform(X):
        return X.dot(self.V)



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

        Returns
        -----------
        L : ndarray of size (d1,d2)
            Low rank matrix, denoised

        S : sparse matrix of size (d1,d2)
            Sparse outliers

        '''

        return (self.U.dot(self.S).dot(self.V), outliers)



