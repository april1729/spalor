from spalor.regularizers import *
import numpy as np
def prox_grad_spca(M,r,alpha1, alpha2, max_iter=100, eps=1e-6):
    (d1,d2)=M.shape
    U=np.zeros((d1,r))
    V=np.zeros((d2,r))
    U_old = np.copy(U)

    for iter in range(0,max_iter):
        U=M.dot(V).dot(np.linalg.pinv(np.transpose(V).dot(V)))
        U=mcp_prox(U, alpha1,2)

        V=np.transpose(M).dot(U).dot(np.linalg.pinv(np.transpose(U).dot(U)))
        V=mcp_prox(V, alpha2,2)

        if np.linalg.norm(U-U_old)< eps:
            break
        U_old=np.copy(U)

    return (U,V)
