import cupy as cp
import numpy as np


def power_system_h_tra1(x, para):
    mac_con = cp.asarray(para[0][0])
    n_mac = mac_con.shape[0]
    Y_gprf = cp.asarray(para[0][5])
    mac_pot = 100 * cp.ones((n_mac, 1)) / cp.asarray(mac_con[:, 2]).reshape(1, -1).T
    Coutput = cp.asarray(para[0][9] - 1)
    s_pos = cp.asarray(para[0][8] - 1)
    xcont = cp.asarray(para[0][10])
    n_col = x.shape[1]
    xaug = cp.zeros((4 * n_mac, n_col))
    for i in range(s_pos.shape[0]):
        xaug[s_pos[i], :] = x[i]
    xaug[cp.setdiff1d(cp.arange(4 * n_mac), s_pos), :] = cp.tile(xcont, (1, n_col))
    if isinstance(Coutput, int) or isinstance(Coutput, float) == 1:
        Y_all = cp.zeros((4, n_col))
    else:
        Y_all = cp.zeros((Coutput.shape[0] * 4, n_col))

    for col in range(n_col):
        dlt = xaug[:n_mac, col].reshape(1, -1).T
        eqp = xaug[2 * n_mac:3 * n_mac, col].reshape(1, -1).T
        edp = xaug[3 * n_mac:4 * n_mac, col].reshape(1, -1).T
        psi_re = cp.sin(dlt) * edp + cp.cos(dlt) * eqp
        psi_im = -cp.cos(dlt) * edp + cp.sin(dlt) * eqp
        psi = psi_re + 1j * psi_im
        iR = cp.real(Y_gprf @ psi)
        iI = cp.imag(Y_gprf @ psi)
        iq = iI * cp.sin(dlt) + iR * cp.cos(dlt)
        id = iR * cp.sin(dlt) - iI * cp.cos(dlt)
        idg = id * mac_pot
        iqg = iq * mac_pot
        eq = eqp - mac_con[:, 6].reshape(1, -1).T * idg
        ed = edp + mac_con[:, 6].reshape(1, -1).T * iqg
        eR = ed * cp.sin(dlt) + eq * cp.cos(dlt)
        eI = eq * cp.sin(dlt) - ed * cp.cos(dlt)
        Y = cp.concatenate((eR[Coutput], eI[Coutput], iR[Coutput], iI[Coutput]))
        Y_all[:, col] = cp.squeeze(Y)

    return cp.asarray(Y_all)
