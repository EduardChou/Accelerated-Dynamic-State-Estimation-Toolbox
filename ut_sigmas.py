import numpy as np

def ut_sigmas(M = None, P = None, c = None):
    A = np.linalg.cholesky(P)
    X = np.concatenate([np.zeros_like(M), A, -A], axis=1)
    X = np.sqrt(c) * X + np.tile(M, (1, X.shape[1]))
    return X
