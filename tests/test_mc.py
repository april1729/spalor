from unittest import TestCase

from spalor.models import MC
from spalor.datasets.synthetic_test_data import mc_test
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

(M, X, y) = mc_test(500, 250, 2, 0.75)
X_train, X_test, y_train, y_test = train_test_split(X.transpose(), y, test_size=0.5)

mc = MC(500, 250, 3)

mc.fit(X_train.transpose(), y_train)

y_pred = mc.predict(X_test.transpose())

print("RFNE:", np.linalg.norm(y_pred - y_test) / np.linalg.norm(y_test))
plt.scatter(y_pred, y_test)
plt.show()


class TestMC(TestCase):
    def test_get_svd(self):
        self.fail()
