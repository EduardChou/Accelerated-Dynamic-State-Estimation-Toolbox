import numpy as np

from ut_transform import ut_transform

def ukf_update1(M = None,P = None,Y = None,h = None,R = None,h_param = None,alpha = None,beta = None,kappa = None,mat = None):

    # Check that all arguments are there


    # Apply defaults

    if mat is None:
        mat = 0

    # Do transform and make the update

    tr_param = [alpha,beta,kappa,mat]
    MU,S,C = ut_transform(M,P,h,h_param,tr_param)
    
    S = S + R
    K = np.dot(C,np.linalg.inv(S))
    M = M + np.dot(K, (Y - MU))
    P = P - np.dot(np.dot(K,S),K.T)

    return M,P,K,MU,S