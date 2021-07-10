from ...algorithms.scpa_algorithms import *


class SPCA():
    def __init__(self, num_components=10, alpha1=0.1, alpha2=1):
        self.num_components=num_components
        self.alpha1=alpha1
        self.alpha2=alpha2
    def fit(self, M):
        (U,V)=hard_thresh_spca(M, self.num_components, self.alpha1, self.alpha2)
        self.U=U
        self.V=V

    def transform(self, X):
        return X.dot(self.V)

    def inverse_transform(self, X):
        return self.U.dot(X)

    def get_params(self):
        return (self.U, self.V)
