from ..regularization import *
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
def hard_thresh_spca(M,r,alpha1, alpha2, max_iter=500, eps=1e-6):
    (d1,d2)=M.shape
    s1=round(d1*alpha1)
    s2=round(d2*alpha2)
    U=np.random.randn(d1,r)
    V=np.random.randn(d2,r)
    U_old = np.copy(U)

    for iter in range(0,max_iter):
        U=M.dot(V).dot(np.linalg.pinv(np.transpose(V).dot(V)))
        for k in range(0,r):
            U[:,k]=sparseProj(U[:,k], s1)
        V=np.transpose(M).dot(U).dot(np.linalg.pinv(np.transpose(U).dot(U)))
        for k in range(0,r):
            V[:,k]=sparseProj(V[:,k], s2)

        if np.linalg.norm(U-U_old)< eps:
            break
        U_old=np.copy(U)

    return (U,V)
