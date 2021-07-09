from spalor.models import RPCA
from spalor.datasets import rpca_test
import numpy as np

L = np.ones([5, 5])
S = np.zeros([5, 5])
S[1, 2] = 10
S[3, 4] = -10
M = L + S 


rpca=RPCA(r=1,sparsity=2)

rpca.fit(M)
(U,V,S)=rpca.get_params()
print("test 1...")
print("L=",U.dot(V.transpose()))
print("S=",np.round(S))

print("test 2...")
(L,S_star)=rpca_test(100,50,2,0.05)
S_star=10*S_star
rpca2=RPCA(r=2, sparsity=250)
rpca2.fit(L+S_star)

(U,V,S)=rpca2.get_params()
print("\t L error:",np.linalg.norm(L-U.dot(V.transpose()), ord='fro')/np.linalg.norm(L, ord='fro'))
print("\t S error:",np.linalg.norm(S-S_star, ord='fro')/np.linalg.norm(S_star, ord='fro'))


print("test 3...")
(L,S_star)=rpca_test(100,500,17,0.03)
S_star=10*S_star
rpca3=RPCA(r=17, sparsity=0.03)
rpca3.fit(L+S_star+0.0*np.random.randn(100,500))

(U,V,S)=rpca3.get_params()
print("\t L error:",np.linalg.norm(L-U.dot(V.transpose()), ord='fro')/np.linalg.norm(L, ord='fro'))
print("\t S error:",np.linalg.norm(S-S_star, ord='fro')/np.linalg.norm(S_star, ord='fro'))
