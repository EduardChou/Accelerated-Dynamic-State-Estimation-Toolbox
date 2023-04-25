import cupy as cp
from ut_weights import ut_weights
from ut_sigmas import ut_sigmas

def ut_transform(M=None, P=None, g=None, g_param=None, tr_param=None):
    # Apply defaults
    if len(tr_param) == 0:
        alpha = None
        beta = None
        kappa = None
        mat = None
        X = None
        w = None
    else:
        alpha = tr_param[0]
        if len(tr_param) >= 2:
            beta = tr_param[1]
        else:
            beta = None
        if len(tr_param) >= 3:
            kappa = tr_param[2]
        else:
            kappa = None
        if len(tr_param) >= 4:
            mat = tr_param[3]
        else:
            mat = None
        if len(tr_param) >= 5:
            X = tr_param[4]
        else:
            X = None
        if len(tr_param) >= 6:
            w = tr_param[5]
        else:
            w = None

    if mat is None:
        mat = 0

    # Calculate sigma points
    if w is not None:
        WM = w[0]
        c = w[2]
        if mat:
            W = w[1]
        else:
            WC = w[1]
    else:
        if mat:
            WM, W, c = ut_mweights(M.shape[1 - 1], alpha, beta, kappa)
            X = ut_sigmas(M, P, c)
            w = [WM, W, c]
        else:
            WM, WC, c = ut_weights(M.shape[1 - 1], alpha, beta, kappa)
            X = ut_sigmas(M, P, c)
            w = [WM, WC, c]

    # Propagate through the function
    if isinstance(g, (int, float, complex, cp.number)):
        Y = g * X
    else:
        Y = None
        for i in cp.arange(0, X.shape[2 - 1]).reshape(-1):
            if (Y is None):
                Y = g(X[:, i].reshape(1, -1).T, g_param)
            else:
                Y = cp.concatenate([Y, g(X[:, i].reshape(1, -1).T, g_param)], axis=1)

    if mat:
        mu = cp.multiply(Y, WM)
        S = cp.multiply(cp.multiply(Y, W), cp.transpose(Y))
        C = cp.multiply(cp.multiply(X, W), cp.transpose(Y))
    else:
        mu = cp.zeros((Y.shape[1 - 1], 1))
        S = cp.zeros((Y.shape[1 - 1], Y.shape[1 - 1]))
        C = cp.zeros((M.shape[1 - 1], Y.shape[1 - 1]))
        WM = cp.asarray(WM)
        WC =cp.asarray(WC)
        M=cp.asarray(M)

        for i in cp.arange(0, X.shape[2 - 1]):
            mu = mu + WM[i] * Y[:, i].reshape(1, -1).T
        for i in cp.arange(0, X.shape[2 - 1]):
            S += WC[i] * cp.outer(Y[:, i].reshape(1, -1).T - mu, Y[:, i].reshape(1, -1).T - mu)
            C += WC[i] * cp.outer(X[:M.shape[0], i].reshape(1, -1).T - M, Y[:, i].reshape(1, -1).T - mu)

    return mu, S, C