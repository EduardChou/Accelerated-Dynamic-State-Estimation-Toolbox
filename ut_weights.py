import numpy as np

def ut_weights(n = None,alpha = None,beta = None,kappa = None):
    # Apply default values

    if alpha is None:
        alpha = 1

    if beta is None:
        beta = 0

    if kappa is None:
        kappa = 3 - n


    # Compute the normal weights

    lambda_ = alpha ** 2 * (n + kappa) - n;
    WM = np.zeros((2 * n + 1,1))
    WC = np.zeros((2 * n + 1,1))
    for j in np.arange(1,2 * n + 1+1).reshape(-1):
        if j == 1:
            wm = lambda_ / (n + lambda_)
            wc = lambda_ / (n + lambda_) + (1 - alpha ** 2 + beta)
        else:
            wm = 1 / (2 * (n + lambda_))
            wc = wm
        WM[j-1] = wm
        WC[j-1] = wc

    c = n + lambda_
    
    return WM,WC,c