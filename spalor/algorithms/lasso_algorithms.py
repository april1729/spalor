from ..regularizers import *


def sparse_prox_grad(XX, Xy, alpha, maxIter=1000, eps=1e-4):

    d=len(Xy)
    n=XX.shape[0]

    w=np.zeros((d,))
    w_old=np.zeros((d,))

    #step_size=2*n/(np.linalg.norm(XX, ord=2))
    step_size=1
    for iter in range(0,maxIter):
        w=mcp_prox(w-(step_size/n)*(XX.dot(w)-Xy), alpha, 2)
    return w


