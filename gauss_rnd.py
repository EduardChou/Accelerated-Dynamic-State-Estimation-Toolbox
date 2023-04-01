import numpy as np

def gauss_rnd(M, S, N=1):
    if N is None:
        N = 1
    L = np.linalg.cholesky(S).T
    if(M.ndim==1):
        X = np.tile(M, (1, N)).reshape(1,-1).T + np.dot(L, np.random.randn(M.shape[0], N))
    else:
        X = np.tile(M, (1, N)).reshape(1,-1).T + np.dot(L, np.random.randn(M.shape[0], M.shape[1]*N))
    return X