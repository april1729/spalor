import numpy as np

def prox_grad_spca(M,r,alpha1, alpha2,prox):
    (d1,d2)=M.shape
    U=np.zeros((d1,r))
    V=np.zeros((d2,r))

    for iter in range(0,50):
        U=M.dot(V).dot(np.linalg.pinv(np.transpose(V).dot(V)))
        U=prox(U, alpha1)

        U=np.transpose(M).dot(U).dot(np.linalg.pinv(np.transpose(U).dot(U)))
        V=prox(V, alpha2)

    return (U,V)