import numpy as np

def power_system_h_tra1(x, para):
    mac_con = para[0][0]
    n_mac = mac_con.shape[0]
    Y_gprf = para[0][5]
    mac_pot = 100 * np.ones((n_mac, 1)) / mac_con[:, 2].reshape(1,-1).T
    Coutput = para[0][9]-1
    s_pos = para[0][8]-1
    xcont = para[0][10]
    n_col = x.shape[1]
    xaug = np.zeros((4 * n_mac, n_col))
    for i in range(s_pos.shape[0]):
        xaug[s_pos[i], :] = x[i]
    xaug[np.setdiff1d(np.arange(4 * n_mac), s_pos), :] = np.tile(xcont, (1, n_col))
    if isinstance(Coutput,int) or isinstance(Coutput,float) == 1:
        Y_all = np.zeros((4, n_col))
    else:
        Y_all = np.zeros((Coutput.shape[0] * 4, n_col))
    
    for col in range(n_col):
        dlt = xaug[:n_mac, col].reshape(1,-1).T
        eqp = xaug[2 * n_mac:3 * n_mac, col].reshape(1,-1).T
        edp = xaug[3 * n_mac:4 * n_mac, col].reshape(1,-1).T
        psi_re = np.sin(dlt) * edp + np.cos(dlt) * eqp
        psi_im = -np.cos(dlt) * edp + np.sin(dlt) * eqp
        psi = psi_re + 1j * psi_im
        iR = np.real(Y_gprf @ psi)
        iI = np.imag(Y_gprf @ psi)
        iq = iI * np.sin(dlt) + iR * np.cos(dlt)
        id = iR * np.sin(dlt) - iI * np.cos(dlt)
        idg = id * mac_pot
        iqg = iq * mac_pot
        eq = eqp - mac_con[:, 6].reshape(1,-1).T * idg
        ed = edp + mac_con[:, 6].reshape(1,-1).T * iqg
        eR = ed * np.sin(dlt) + eq * np.cos(dlt)
        eI = eq * np.sin(dlt) - ed * np.cos(dlt)
        Y = np.concatenate((eR[Coutput], eI[Coutput], iR[Coutput], iI[Coutput]))
        Y_all[:, col] = np.squeeze(Y)
        
    return Y_all