import numpy as np
import time as tm
from scipy.io import loadmat, savemat

from gauss_rnd import gauss_rnd
from power_system_f_tra import power_system_f_tra
from power_system_f_tra1 import power_system_f_tra1
from power_system_h_tra import power_system_h_tra
from power_system_h_tra1 import power_system_h_tra1
from ukf_predict1 import ukf_predict1
from ukf_update1 import ukf_update1

def test_UKFGPS_SRUKF():
    np.random.seed(0)
    # import data files
    system = 3 # 3 for 3-machine system; 48 for 48-machine system
    if system == 3:
        x0 = loadmat('x0_3m.mat')['x0']
        x00 = loadmat('x00_3m.mat')['x00']
        para_pf2 = loadmat('para_pf2_3m.mat')['para_pf2']
        mac_tra_idx = para_pf2[0][6] - 1
        mac_em_idx = para_pf2[0][7] - 1
        n_mac = 3
        n_pmu = 1
        sensorpos_opt_det = np.array([3])
    elif system == 48:
        x0 = loadmat('x0_48m.mat')['x0']
        x00 = loadmat('x00_48m.mat')['x00']
        para_pf2 = loadmat('para_pf2_48m.mat')['para_pf2']
        mac_tra_idx = para_pf2[0][6] - 1
        mac_em_idx = para_pf2[0][7] - 1
        n_mac = 48
        Placement_48m = loadmat('Placement_48m.mat')['x1_basic']
        opt_det = Placement_48m
        n_pmu = 24
        sensorpos_opt_det = np.nonzero(opt_det[n_pmu,:])[0]+1
    index_record = np.zeros((7,8,1))
    time_record_ukf_gps_record = np.array([])
    time_r = np.array([])
    idx = 0
    for i in range(n_pmu):
        idx += 1
        n_tra = mac_tra_idx.shape[0]
        n_s = 4*n_tra + 2*mac_em_idx.shape[0]
        noisedltbd = 0.5*np.pi/180
        noisewbd = 1e-3*2*np.pi*60
        noiseeqpbd = 1e-3
        noiseedpbd = 1e-3
        P = np.block([[noisewbd**2*np.eye(n_mac), np.zeros((n_mac, n_mac)), np.zeros((n_mac, n_tra)), np.zeros((n_mac, n_tra))],[np.zeros((n_mac, n_mac)), noisedltbd**2*np.eye(n_mac), np.zeros((n_mac, n_tra)), np.zeros((n_mac, n_tra))],[np.zeros((n_tra, n_mac)), np.zeros((n_tra, n_mac)), noiseeqpbd**2*np.eye(n_tra), np.zeros((n_tra, n_tra))],[np.zeros((n_tra, n_mac)), np.zeros((n_tra, n_mac)), np.zeros((n_tra, n_tra)), noiseedpbd**2*np.eye(n_tra)]])
        index,time_record_ukf_gps,needTime,time = testcase(n_mac,n_s,x00,x0,sensorpos_opt_det,para_pf2,P)
        index_record[:,:,idx-1] = index
        time_r = np.append(time_r, time)
        time_record_ukf_gps_record = np.append(time_record_ukf_gps_record, time_record_ukf_gps)
        savemat(f'.\index_KF_{n_mac}m.mat', {'index_record': index_record, 'index': index, 'time_r': time_r, 'time_record_ukf_gps': time_record_ukf_gps, 'needTime': needTime, 'time': time})
        # In 'index_record.mat', each row corresponds to EKF, UKF-schol, UKF-GPS,
        # SR-UKF, UKF-kappa, UKF-modified, UKF-DeltaQ

def testcase(n_mac, n_s, x00, x0, sensorpos_opt_det, para, P):
    # simulate
    t0 = 0
    Tfinal = 10
    sensorfreq = 120
    tsequence = np.arange(t0, Tfinal+1./sensorfreq, 1./sensorfreq)
    
    # solve ODE
    para[0][4] = 1./sensorfreq
    s_pos = para[0][8]
    states_nonoise = np.zeros((tsequence.size, 4*n_mac))
    states_nonoise[0,:] = x0.reshape(1,-1)

    for i in range(1, tsequence.size):
        states_nonoise[i,:] = power_system_f_tra(states_nonoise[i-1,:].reshape(1,-1).T, para).reshape(1,-1)
    states_nonoise = states_nonoise[:, np.squeeze(np.asarray(s_pos-1))]
    
    Q = np.diag((0.1 * np.max(np.abs(np.diff(states_nonoise,axis=0)),axis=0)) ** 2)
    states = np.zeros((tsequence.size, 4*n_mac))
    states[0,:] = x0.reshape(1,-1)

    for i in range(1, tsequence.size):
        states[i,:] = power_system_f_tra(states[i-1,:].reshape(1,-1).T, para).reshape(1,-1)
        processNoise = gauss_rnd(np.zeros(n_s), Q)
        states[i,np.squeeze(np.asarray(s_pos-1))] += np.squeeze(processNoise.reshape(1,-1))
    
    # set initial value
    M = x00

    # dynamic state estimation
    para[0][4] = 1./60
    index, time_record_ukf_gps, needTime, time = ekf_ukf(n_mac, sensorpos_opt_det, states, M, P, Q, para, tsequence, Tfinal)

    return index, time_record_ukf_gps, needTime, time

def ekf_ukf(n_mac, sensorpos, states, M0, P0, Q, para, tsequence, Tfinal):
    # measurements
    sensorfreq = 1 / para[0][4]
    ratio = (len(tsequence) - 1) / (sensorfreq * Tfinal)
    tsequence = tsequence[::int(ratio)]
    n_sensor = sensorpos.shape[0]

    # create measurements with noise
    f_func = power_system_f_tra1
    h_func = power_system_h_tra1
    para = np.insert(para, 9, 0, axis=1)
    para[0][9] = sensorpos.reshape(1,-1).T
    states = states[::int(ratio), :]
    Y_real = power_system_h_tra(states[0, :].reshape(1,-1).T, para)
    for i in range(1,states.shape[0]):
        Y_real = np.insert(Y_real,i,values=np.squeeze(np.asarray(power_system_h_tra(states[i, :].reshape(1,-1).T, para))),axis=1)
    R = np.diag(0.01**2 * np.ones((4 * n_sensor)))
    Y = Y_real.copy()
    for i in range(4 * n_sensor):
        Y[i, :] = Y[i, :] + np.sqrt(R[i, i]) * np.random.randn(Y.shape[1])
    # perform rest of the calculations here
    # ...
    
    # estimate with UKF-GPS
    s_pos = para[0][8]
    n_s = s_pos.shape[0]
    U_MM_gps = np.zeros((n_s,Y.shape[1]))
    U_MM_gps[:,0] = M0[np.squeeze(np.asarray(s_pos-1)),:].flatten()
    U_PP_gps = np.zeros((n_s,n_s,Y.shape[1]))
    M =  M0[np.squeeze(np.asarray(s_pos-1)),:].reshape(1,-1).T
    para = np.insert(para, 10, 0 ,axis =1)
    para[0][10]=M0[np.squeeze(np.setdiff1d(np.arange(4*n_mac), s_pos-1)),:]
    P = P0
    tPD = 0
    flag_p = 0
    num_solve = 0
    iteration_r = 0
    norm_r = 0
    allconverged = 1
    needTime = []
    tstart1 = tm.time()
    for k in range(1,Y.shape[1]):
        M,P,temp_ = ukf_predict1(M,P,f_func,Q,para)
        p = np.linalg.matrix_rank(P)
        if p < P.shape[0]:
            num_solve += 1
            needtm.append([tsequence[k], 1])
            flag_p = 1
            tstart2 = tm.time()
            P_tmp = P
            P,normF,iterations,converged = nearPD_matlab(P_tmp)
            if converged == 0:
                allconverged = 0
            iteration_r += iterations
            norm_r += normF
            tPD += tm.time() - tstart2
        M,P,temp_,temp__,temp___ = ukf_update1(M,P,Y[:,k].reshape(1,-1).T,h_func,R,para)
        p = np.linalg.matrix_rank(P)
        if p < P.shape[0]:
            num_solve += 1
            needtm.append([tsequence[k], 2])
            flag_p = 1
            P_tmp = P
            tstart2 = tm.time()
            P,normF,iterations,converged = nearPD_matlab(P_tmp)
            if converged == 0:
                allconverged = 0
            iteration_r += iterations
            norm_r += normF
            tPD += tm.time() - tstart2
        U_MM_gps[:,k] = np.squeeze(M)
        U_PP_gps[:,:,k] = P
    time_ukfgps = tm.time() - tstart1
    time_record_ukf_gps = [num_solve, tPD, np.float64(iteration_r)/num_solve, np.float64(norm_r)/num_solve, allconverged, flag_p]
    
    time = time_ukfgps

    index3 = error(states,U_MM_gps,para,tsequence,s_pos,n_mac)
    
    index = index3

    return index,time_record_ukf_gps,needTime,time
    
def error(states, U_MM, para, tsequence, s_pos, n_mac):
    U_MM1 = U_MM.T
    states = states[:, np.squeeze(np.asarray(s_pos-1))]
    dltX = U_MM1 - states
    index_delta = 0
    index_omega = 0
    index_eqp = 0
    index_edp = 0
    mac_tra_idx = para[0][6]
    for i in range(len(tsequence)):
        index_delta += 1/n_mac * np.sum((dltX[i,0:n_mac])**2)
        index_omega += 1/n_mac * np.sum((dltX[i,n_mac:2*n_mac])**2)
        index_eqp += np.float64(1)/len(mac_tra_idx) * np.float64(np.sum((dltX[i,2*n_mac:2*n_mac+len(mac_tra_idx)])**2))
        index_edp += np.float64(1)/len(mac_tra_idx) * np.sum((dltX[i,2*n_mac+len(mac_tra_idx):])**2)
    index_delta = np.sqrt(index_delta /(len(tsequence)))
    index_omega = np.sqrt(index_omega /(len(tsequence)))
    index_eqp = np.sqrt(index_eqp /(len(tsequence)))
    index_edp = np.sqrt(index_edp /(len(tsequence)))
    # number of convergent states
    index_num_dlt = np.zeros(n_mac)
    index_num_omg = np.zeros(n_mac)
    index_num_eqp = np.zeros(len(mac_tra_idx))
    index_num_edp = np.zeros(len(mac_tra_idx))
    t_4 = np.where(tsequence==9)[0][0]
    for i in range(n_mac):
        if np.all(np.abs(dltX[t_4:,i]) < np.abs(states[t_4:,i])*0.01):
            index_num_dlt[i] = 1
        if np.all(np.abs(dltX[t_4:,n_mac+i]) < np.abs(states[t_4:,n_mac+i])*0.01):
            index_num_omg[i] = 1
    for i in range(len(mac_tra_idx)):
        if np.all(np.abs(dltX[t_4:,2*n_mac+i]) < np.abs(states[t_4:,2*n_mac+i])*0.01):
            index_num_eqp[i] = 1
        if np.all(np.abs(dltX[t_4:,2*n_mac+len(mac_tra_idx)+i]) < np.abs(states[t_4:,2*n_mac+len(mac_tra_idx)+i])*0.01):
            index_num_edp[i] = 1
    index = np.concatenate(([index_delta, index_omega, index_eqp, index_edp], [np.sum(index_num_dlt), np.sum(index_num_omg), np.sum(index_num_eqp), np.sum(index_num_edp)]))
    
    return index