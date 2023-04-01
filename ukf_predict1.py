import numpy as np
from ut_transform import ut_transform

def ukf_predict1(M = None,P = None,f = None,Q = None,f_param = None,alpha = None,beta = None,kappa = None,mat = None):
    # Apply defaults

    if f is None:
        f = np.eye(M.shape[1-1])

    if Q is None:
        Q = np.zeros((M.shape[1-1],M.shape[1-1]))

    if mat is None:
        mat = 0

    # Do transform
    # and add process noise

    tr_param = [alpha,beta,kappa,mat]
    M,P,D = ut_transform(M,P,f,f_param,tr_param)
    P = P + Q
    return M,P,D