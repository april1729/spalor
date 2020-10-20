from spalor.regularizers import *


def sparse_prox_grad(XX, Xy, alpha, maxIter=100, eps=1e-6):

    d=len(Xy)
    n=len(XX)

    w=np.zeros((d,))
    w_old=np.zeros((d,))

    step_size=2*n/(np.linalg.norm(XX, 2))
    for iter in range(0,maxIter):
        w=mcp_prox(w-(step_size/n)*(XX.dot(w)-Xy), step_size*alpha, 2)
        if np.linalg.norm(w-w_old)<eps:
            break
        w_old=np.copy(w)
    return w


