import cupy as cp

def ut_sigmas(M = None, P = None, c = None):
    P=cp.asarray(P)
    A = cp.linalg.cholesky(P)
    X = cp.concatenate([cp.zeros_like(cp.asarray(M)), A, -A], axis=1)
    X = cp.sqrt(c) * X + cp.tile(cp.asarray(M), (1, X.shape[1]))
    return X