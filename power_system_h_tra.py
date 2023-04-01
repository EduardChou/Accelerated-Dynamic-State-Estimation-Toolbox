import numpy as np

def power_system_h_tra(x, para):
    mac_con = para[0][0]
    n_mac = mac_con.shape[0]
    Y_gprf = para[0][5]
    mac_pot = 100 * np.ones((n_mac, 1)) / mac_con[:, 2].reshape(1,-1).T
    Coutput = np.squeeze(np.asarray(para[0][9]-1))
    
    dlt = x[0:n_mac]
    omg = x[n_mac:2*n_mac]
    eqp = x[2*n_mac:3*n_mac]
    edp = x[3*n_mac:4*n_mac]
    
    psi_re = np.sin(dlt)*edp + np.cos(dlt)*eqp
    psi_im = -np.cos(dlt)*edp + np.sin(dlt)*eqp
    psi = psi_re + 1j*psi_im
    It = np.dot(Y_gprf, psi)
    iR = np.real(It)
    iI = np.imag(It)
    iq = iI*np.sin(dlt) + iR*np.cos(dlt)
    id = iR*np.sin(dlt) - iI*np.cos(dlt)
    idg = id*mac_pot
    iqg = iq*mac_pot
    eq = eqp - mac_con[:,6].reshape(1,-1).T*idg
    ed = edp + mac_con[:,6].reshape(1,-1).T*iqg
    eR = ed*np.sin(dlt)+eq*np.cos(dlt)
    eI = eq*np.sin(dlt)-ed*np.cos(dlt)
    Y = np.vstack((eR[Coutput,:], eI[Coutput,:], iR[Coutput,:], iI[Coutput,:]))
    return Y