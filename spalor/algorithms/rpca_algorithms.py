from ..regularization import *

def altProjNiave(M, r, s, fTol=1e-10, maxIter=100):
    res = np.inf
    S = np.zeros(M.shape)
    L = np.zeros(M.shape)
    for k in range(0, maxIter):
        S = sparseProj(M - L, s)
        L = lowRankProj(M - S, r)

        res0 = res
        res = np.linalg.norm(M - (L + S), ord='fro')
        if (res0 - res) / res < fTol:
            break
    return (L, S)

def altSoftThresh(M, beta, fTol=1e-6, maxIter=1000):
    sqrtN = np.sqrt(min(M.shape))
    res = np.inf
    L = np.zeros(M.shape)
    S = np.zeros(M.shape)
    # TODO: Make beta increase with every iteration
    for k in range(0, maxIter):
        S = sparseSoftThresholding(M - L, 1 / (sqrtN * beta))
        L = lowRankSoftThresholding(M - S, 1 / beta)
        # TODO: fix the stopping criteria in altSoftThresh
    #  res0 = res
    # res = np.linalg.norm(M-(L+S), ord='fro')
    # if (res0-res)/res < fTol:
    #     break
    return L, S

def altAdaptiveThresh(M, beta, fTol=1e-6, maxIter=1000):
    sqrtN = np.sqrt(min(M.shape))
    res = np.inf
    L = np.zeros(M.shape)
    S = np.zeros(M.shape)
    # TODO: Make beta increase with every iteration
    for k in range(0, maxIter):
        S = sparseSoftThresholding(M - L, 1 / (sqrtN * beta))
        L = lowRankSoftThresholding(M - S, 1 / beta)
        # TODO: fix the stopping criteria in altSoftThresh
    #  res0 = res
    # res = np.linalg.norm(M-(L+S), ord='fro')
    # if (res0-res)/res < fTol:
    #     break
    return L, S

def altProj(M, r=None, eps=1e-5, beta=None):
    n = min(M.shape)

    if beta is None:
        beta = 10 / n ** 0.5

    threshold = beta * singularValue(M, 0)

    if r is None:
        r = n-2
    print(r)
    L = np.zeros(M.shape)
    S = sparseHardThresholding(M - L, threshold)

    for k in range(0, r):
        T = int(round(10 * np.log(n * beta * np.linalg.norm(M - S, ord=2) / eps)))
        print(k, T)

        for t in range(0, T):
            threshold = beta * (singularValue(M - S, k + 1) + (1 / 2) ** t * singularValue(M - S, k))
            L = lowRankProj(M - S, k + 1);
            S = sparseHardThresholding(M - L, threshold)

        print(k, singularValue(L, k))
        if beta * (singularValue(L, k)) <= eps / (2 * n):
            return L, S
    return L, S

def altNonconvexThresh(M, beta, lambdas, maxIter=1000):

    L = np.zeros(M.shape)
    S = np.zeros(M.shape)

    for k in range(0, maxIter):
        S = sparseProxThresholding(M - L, proxFuncS)
        L = lowRankProxThresholding(M - S, proxFuncL)

    return L,S

if __name__ == "__main__":
    L = np.ones([10, 10])
    S = np.zeros([10, 10])
    S[1, 2] = 10
    S[5, 9] = 10
    S[3, 5] = -10
    M = L + S

    (Lproj, Sproj) = altProj(M)
    print(Lproj)
    print(np.round(Sproj))