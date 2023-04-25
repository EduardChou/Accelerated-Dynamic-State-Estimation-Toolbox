import cupy as cp
from ut_transform import ut_transform

def ukf_update1(M=None, P=None, Y=None, h=None, R=None, h_param=None, alpha=None, beta=None, kappa=None, mat=None):

    # Check that all arguments are there

    # Apply defaults
    if mat is None:
        mat = 0

    # Do transform and make the update
    tr_param = [alpha, beta, kappa, mat]
    MU, S, C = ut_transform(cp.asnumpy(M), cp.asnumpy(P), h, h_param, tr_param)
    MU, S, C = cp.asarray(MU), cp.asarray(S), cp.asarray(C)
    R=cp.asarray(R)
    M=cp.asarray(M)
    Y = cp.asarray(Y)
    P=cp.asarray(P)

    S = S + R
    K = cp.dot(C, cp.linalg.inv(S))
    M = M + cp.dot(K, (Y - MU))
    P = P - cp.dot(cp.dot(K, S), K.T)

    return cp.asarray(M), cp.asarray(P), cp.asarray(K), MU, S
