import numpy as np

def calTemp(dlt, eqp, edp, mac_con, Y_gprf, mac_pot):
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
    Te = (ed*id + eq*iq)*mac_pot
    return iqg, idg, eq, ed, Te

def power_system_f_tra1(x, para):
    mac_con = para[0][0]
    n_mac = mac_con.shape[0]
    xd = mac_con[:,5].reshape(1,-1).T
    xdp = mac_con[:,6].reshape(1,-1).T
    Td0p = mac_con[:,8].reshape(1,-1).T
    xq = mac_con[:,10].reshape(1,-1).T
    xqp = mac_con[:,11].reshape(1,-1).T
    Tq0p = mac_con[:,13].reshape(1,-1).T
    H2 = 2 * mac_con[:,15].reshape(1,-1).T
    wR = para[0][1]
    Pm = para[0][2]
    Efd = para[0][3]
    dt = para[0][4]
    Y_gprf = para[0][5]
    mac_tra_idx = para[0][6]-1
    mac_em_idx = para[0][7]-1
    s_pos = para[0][8]-1
    xcont = para[0][10]
    mac_pot = 100 * np.ones((n_mac, 1)) / mac_con[:, 2].reshape(1,-1).T
    
    n_col = x.shape[1]
    xaug = np.zeros((4*n_mac, n_col))
    xaug[np.squeeze(np.asarray(s_pos)),:] = x
    xaug[np.setdiff1d(np.arange(4*n_mac), s_pos),:] = np.tile(xcont, (n_col, 1))

    x_n_all = np.zeros((len(s_pos), n_col))
    
    for col in range(n_col):
        dlt = xaug[0:n_mac,col].reshape(1,-1).T
        omg = xaug[n_mac:2*n_mac,col].reshape(1,-1).T
        eqp = xaug[2*n_mac:3*n_mac,col].reshape(1,-1).T
        edp = xaug[3*n_mac:,col].reshape(1,-1).T

        iqg, idg, eq, ed, Te = calTemp(dlt, eqp, edp, mac_con, Y_gprf, mac_pot)
        ddlt = omg - wR
        domg = wR*(Pm - Te - mac_con[:,16].reshape(1,-1).T*(omg-wR)/wR)/H2
        deqp = np.zeros((n_mac,1))
        dedp = np.zeros((n_mac,1))
        deqp[mac_em_idx] = 0
        dedp[mac_em_idx] = 0
        deqp[mac_tra_idx] = (Efd[mac_tra_idx] - eqp[mac_tra_idx] - (xd[mac_tra_idx]-xdp[mac_tra_idx])*idg[mac_tra_idx])/Td0p[mac_tra_idx]
        dedp[mac_tra_idx] = (-edp[mac_tra_idx] + (xq[mac_tra_idx] - xqp[mac_tra_idx])*iqg[mac_tra_idx])/Tq0p[mac_tra_idx]
        dstate = np.concatenate((ddlt, domg, deqp, dedp))
        x_n1 = xaug[:,col].reshape(1,-1).T + dt*dstate.reshape((-1,1))

        dlt1 = x_n1[0:n_mac]
        omg1 = x_n1[n_mac:2*n_mac]
        eqp1 = x_n1[2*n_mac:3*n_mac]
        edp1 = x_n1[3*n_mac:]
        iqg1, idg1, eq1, ed1, Te1 = calTemp(dlt1, eqp1, edp1, mac_con, Y_gprf, mac_pot)
        ddlt1 = omg1 - wR
        domg1 = wR*(Pm - Te1 - mac_con[:,16].reshape(1,-1).T*(omg1-wR)/wR)/H2
        deqp1 = np.zeros((n_mac,1))
        dedp1 = np.zeros((n_mac,1))
        deqp1[mac_em_idx] = 0
        dedp1[mac_em_idx] = 0
        deqp1[mac_tra_idx] = (Efd[mac_tra_idx] - eqp1[mac_tra_idx] - (xd[mac_tra_idx]-xdp[mac_tra_idx])*idg1[mac_tra_idx])/Td0p[mac_tra_idx]
        dedp1[mac_tra_idx] = (-edp1[mac_tra_idx] + (xq[mac_tra_idx] - xqp[mac_tra_idx])*iqg1[mac_tra_idx])/Tq0p[mac_tra_idx]
        dstate1 = np.concatenate((ddlt1, domg1, deqp1, dedp1))
        x_n = xaug[:,col].reshape(1,-1).T + dt/2*(dstate + dstate1).reshape((-1,1))
        x_n = x_n[s_pos]
        x_n_all[:,col] = np.squeeze(np.asarray(x_n))
    
    return x_n_all
