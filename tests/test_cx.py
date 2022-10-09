from unittest import TestCase

import numpy as np
from spalor.datasets import rand_low_rank_mat
from spalor.models import CUR
from spalor.algorithms.cx_algorithms import group_sparse_regression_CX

M = rand_low_rank_mat(10, 7, 3)

cur = CUR(M, n_rows=3, n_cols=4, r=3)
cur.fit()
(C, U, R) = cur.get_params()

print("test 1 RFNE: ", np.linalg.norm(M - C.dot(U).dot(R)) / np.linalg.norm(M))

M = rand_low_rank_mat(30, 20, 4)

cur = CUR(M, n_rows=30, n_cols=5, r=4)
cur.fit()
(C, U, R) = cur.get_params()

print("test 2 RFNE: ", np.linalg.norm(M - C.demaot(U).dot(R)) / np.linalg.norm(M))

M = rand_low_rank_mat(30, 20, 4)

(C, X, cols) = group_sparse_regression_CX(M, 10)
print("test 3 RFNE: ", np.linalg.norm(M - C.dot(X)) / np.linalg.norm(M))


class TestCX(TestCase):
    def test_fit_from_svd(self):
        self.fail()
