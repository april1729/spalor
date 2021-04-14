import numpy as np
from spalor.models import SPCA
from spalor.regularization import *
from matplotlib import pyplot as plt
U=np.random.randn(60,2)
V=np.random.randn(40,2)
M=U.dot(V.transpose())

spca=SPCA(num_components=10,alpha1=1,alpha2=0.2)
spca.fit(M)

(U,V)=spca.get_params()
print(V)
print(np.linalg.norm(U.dot(V.transpose())-M,ord='fro')/np.linalg.norm(M,ord='fro'))

plt.imshow(abs(V))
plt.show()