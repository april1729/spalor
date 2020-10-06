from SpaLoR.MC.algorithms import *
import numpy as np


class MC:
    def __init__(self, m, n, rank):
        self.m = m
        self.n = n
        self.rank = rank
        self.user_means = np.zeros(m)
        self.user_std = np.zeros(m)

    def fit(self, X, y):
        y_fit = np.zeros(len(y))
        for user in range(0, self.m):
            idx = np.where(X[0, :] == user)
            self.user_means[user] = np.mean(y[idx])
            self.user_std[user] = np.std(y[idx])
            y_fit[idx] = (y[idx] - self.user_means[user]) / self.user_std[user]
        (U, V) = lmafit_mc_adp(self.m, self.n, self.rank, X, y_fit)
        self.U = U
        self.V = V

    def predict(self, X):
        y = partXY(self.U, self.V, X)

        for user in range(0, self.m):
            idx = np.where(X[0, :] == user)
            y[idx] = (y[idx] * self.user_std[user]) + self.user_means[user]

        return y
