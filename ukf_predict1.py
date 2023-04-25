import cupy as cp
import numpy as np
from ut_transform import ut_transform

def ukf_predict1(M=None, P=None, f=None, Q=None, f_param=None, alpha=None, beta=None, kappa=None, mat=None):
    # Apply defaults
    if f is None:
        f = cp.eye(M.shape[1-1])

    if Q is None:
        Q = cp.zeros((M.shape[1-1], M.shape[1-1]))

    if mat is None:
        mat = 0

    # Convert input matrices to Cupy arrays
    M_cupy = cp.asarray(M)
    P_cupy = cp.asarray(P)
    Q_cupy = cp.asarray(Q)

    # Do transform
    # and add process noise

    tr_param = [alpha, beta, kappa, mat]
    M_cupy, P_cupy, D_cupy = ut_transform(M_cupy, P_cupy, f, f_param, tr_param)
    P_cupy = P_cupy + Q_cupy

    # Convert output matrices back to NumPy arrays
    M_numpy = cp.asnumpy(M_cupy)
    P_numpy = cp.asnumpy(P_cupy)
    D_numpy = cp.asnumpy(D_cupy)

    return M_numpy, P_numpy, D_numpy
