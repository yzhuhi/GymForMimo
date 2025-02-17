import numpy as np
import copy
from scipy.linalg import sqrtm
import os, sys
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径
'''
    # Simulation parameters
    N = 128  # Number of antennas
    L = 4    # Number of scatterers in each cluster
    K = 100  # Number of users
    v_min = -20  # Minimum velocity (km/hr)
    v_max = 40   # Maximum velocity (km/hr)
    d = 0.5 * 3e8 / 60e9  # Antenna element spacing
    lambda_ = 3e8 / 60e9  # Wavelength
    sigma_s = 10  # Angular spread standard deviation
    R_min = 20  # Minimum distance from BS (m)
    R_max = 200  # Maximum distance from BS (m)
    phi_k_range = [-60, 60]  # Azimuth angle range (degrees)

    # Simulation time parameters
    Tframe = 6e-3  # Frame length
    fs = 1e5  # Data sampling rate
    t = np.arange(0, Tframe, 1/fs)  # Time vector

    # Initialize result arrays
    H_kt = np.zeros((len(t), K, N), dtype=np.complex128)
    R_k = np.zeros((K, N, N), dtype=np.complex128)
    users_coord = []
    
    # Generate channel model
    for i, ti in enumerate(t):
'''
# 相关系数矩阵Rtx,Rrx
def correlationmatrix(nr, nt, pt, pr):
    Rtx = np.zeros((nt, nt))
    Rrx = np.zeros((nr, nr))

    # Generate transmit correlation matrix
    for i in range(nt):
        for j in range(nt):
            distance = abs(i - j)
            Rtx[i, j] = pt ** distance

    # Generate receive correlation matrix
    for i in range(nr):
        for j in range(nr):
            distance = abs(i - j)
            Rrx[i, j] = pr ** distance

    return Rtx, Rrx

def channel(time_step, user_coordinate, 
            K = 20, d=0.5, fs=1e5, L=4, N=16, v_min = -5.5, v_max = 11.1, R_min=20, R_max=200, lambda_=3e8 / 60e9):
    
    H_kt = np.zeros((K, N), dtype=np.complex128)
    R_k = np.zeros((K, N, N), dtype=np.complex128)
    
    
    for k in range(K):
        v_horizontal = np.random.uniform(v_min, v_max)
        v_vertical = np.random.uniform(v_min, v_max)

        # 计算合速度和速度方向角
        v = np.sqrt(v_horizontal**2 + v_vertical**2)
        phi_v = np.arctan2(v_vertical, v_horizontal)

        # 判断相对速度的符号
        if -np.pi/2 <= phi_v <= np.pi/2:
            v_relative = v
        else:
            v_relative = -v

        # 计算当前时刻的最大多普勒频率
        fm = v_relative / lambda_

        AOA = np.random.laplace(0, 10, L) % (2 * np.pi)  # 使用 Laplacian 分布生成 AOAs
        AOD = np.random.laplace(0, 10, L) % (2 * np.pi)  # 使用 Laplacian 分布生成 AODs

        alpha = np.sqrt(N/L)
        for l in range(L):
            fD = fm * np.cos(AOA[l])  # 使用 AOA 生成相应的 Doppler shift
            beta_kl = np.random.normal(0, 1) + 1j * np.random.normal(0, 1)
            ak_theta_l = np.exp(-1j * 2 * np.pi * np.arange(N) * d / lambda_ * np.sin(AOD[l]))
            H_kt[k, :] += alpha * beta_kl * np.exp(1j * 2 * np.pi * fD * time_step) * ak_theta_l

        # 记录用户位置信息
        if time_step == 0:  # 初始位置
            x = np.zeros((K, 1))
            y = np.zeros((K, 1))
            user_coordinates = user_coordinate
            x[k] = np.random.uniform(R_min, R_max) * np.cos(phi_v)
            y[k] = np.random.uniform(R_min, R_max) * np.sin(phi_v)
            user_coordinates.append([x[k], y[k]])
            # user_position = np.array(x[k],y[k])
        else:  # 更新位置
            user_coordinates = user_coordinate
            r = v * (1/fs)  # 移动距离
            theta = np.arctan2(v_vertical, v_horizontal)  # 移动方向
            user_coordinates[k][0] += r * np.cos(theta)
            user_coordinates[k][1] += r * np.sin(theta)

            dist_BaseToUser = np.sqrt((user_coordinates[k][0]) ** 2 + (user_coordinates[k][1]) ** 2)
            
            if dist_BaseToUser > R_max:
                user_coordinates[k][0] = user_coordinates[k][0] - (2 * r * np.cos(theta))
                user_coordinates[k][1] = user_coordinates[k][1] - (2 * r * np.sin(theta))
            elif dist_BaseToUser < R_min:
                user_coordinates[k][0] = user_coordinates[k][0] - (2 * r * np.cos(theta))
                user_coordinates[k][1] = user_coordinates[k][1] - (2 * r * np.sin(theta))
            # 处理边界情况
            # if user_coordinates[k][0] < R_min:
            #     user_coordinates[k][0] = R_min + (R_min - user_coordinates[k][0])
            # elif user_coordinates[k][0] > R_max:
            #     user_coordinates[k][0] = R_max - (user_coordinates[k][0] - R_max)
            # if user_coordinates[k][1] < R_min:
            #     user_coordinates[k][1] = R_min + (R_min - user_coordinates[k][1])
            # elif user_coordinates[k][1] > R_max:
            #     user_coordinates[k][1] = R_max - (user_coordinates[k][1] - R_max)
                
            # user_position = np.array(user_coordinates[k][0],user_coordinates[k][1])
            
    # Calculate channel covariance matrix
    for k in range(K):
        for l in range(L):
            ak_theta_l = np.exp(-1j * 2 * np.pi * np.arange(N) * d / lambda_ * np.sin(AOD[l]))
            R_k[k, :, :] += alpha**2 * np.outer(ak_theta_l, np.conj(ak_theta_l))    # Calculate channel covariance matrix
        for k in range(K):
            for l in range(L):
                ak_theta_l = np.exp(-1j * 2 * np.pi * np.arange(N) * d / lambda_ * np.sin(AOD[l]))
                R_k[k, :, :] += alpha**2 * np.outer(ak_theta_l, np.conj(ak_theta_l))
            
    return H_kt, R_k, user_coordinates

# 相关衰落瑞利信道
def generate_data(t_list, nt, users, nr, pt, pr, fs, random_seed=1234, downlink=True):
    if random_seed is not None:
        np.random.seed(random_seed)
        
    users_coord = []
    users_cor = []
    n_observation = len(t_list)
    
    if downlink:
        H_array = np.zeros((n_observation, users, nt), dtype=np.complex128)
        H_split_array = np.zeros((n_observation, 2, users, nt))
    else:
        H_array = np.zeros((n_observation, nt, users), dtype=np.complex128)
        H_split_array = np.zeros((n_observation, 2, nt, users))
        
    for id, n_obs in enumerate(t_list):
        
        np.random.seed(random_seed + id)
        H, R_k, user_coordinates = channel(n_obs, users_cor, K=users, fs=fs, N=nt)
        users_cor = copy.deepcopy(user_coordinates)
        if downlink:
        # Downlink 相关衰落信道矩阵 仅考虑基站端
            users_coord.append(copy.deepcopy(user_coordinates))
            # H_array = np.zeros((n_observation, users, nt), dtype=np.complex128)
            # H_split_array = np.zeros((n_observation, 2, users, nt))
            Rtx, Rrx = correlationmatrix(users, nt, pt, pr)
            H = H @ sqrtm(Rtx)

            # 实部虚部分离信道矩阵
            H_split = np.zeros((2, H.shape[0], H.shape[1]))
            H_split[0, :, :] = np.real(H)
            H_split[1, :, :] = np.imag(H)
            # 存储 H 和 H_split 到数组中
            H_array[id] = H
            H_split_array[id] = H_split
        else:
            # uperlink 相关衰落信道矩阵 仍仅考虑基站端
            users_coord.append(copy.deepcopy(user_coordinates))
            # H_array = np.zeros((n_observation, nt, users), dtype=np.complex128)
            # H_split_array = np.zeros((n_observation, 2, nt, users))
            Rtx, Rrx = correlationmatrix(nt, users, pt, pr)
            H = sqrtm(Rrx) @ H.T.conjugate() 
            # 实部虚部分离信道矩阵
            H_split = np.zeros((2, H.shape[0], H.shape[1]))
            H_split[0, :, :] = np.real(H)
            H_split[1, :, :] = np.imag(H)
            # 存储 H 和 H_split 到数组中
            H_array[id] = H
            H_split_array[id] = H_split
        
    return H_array, H_split_array, users_coord

# 生成数据
if __name__ == '__main__':
    
    n_observation = 700 # 信道矩阵个数
    nt = 16  # 基站天线数
    users = 20  # 用户总数
    users1 = 30
    nr = 1  # 单用户天线数
    pt = 0.2  # 相关因子
    pr = 0.2
    # Simulation time parameters
    Tframe = 700/6  # Frame length
    fs = 6  # Data sampling rate
    t = np.arange(0, Tframe, 1/fs)  # Time vector

    H_array, H_split_array, user_coordinates = generate_data(t, nt, users1, nr, pt, pr, fs, downlink=False)

    # 保存数据到文件
    test_path_20 = curr_path + '/dataset/train/Doppler/uplink_30.npz'
    # test_path_30 = curr_path + '/dataset/train/HighV/uplink_30.npz'
    os.makedirs(os.path.dirname(test_path_20), exist_ok=True)
    # os.makedirs(os.path.dirname(test_path_30), exist_ok=True)

    np.savez(test_path_20, H_array=H_array, H_split_array=H_split_array, user_coordinates=user_coordinates)
    # np.savez(test_path_30, H_array=H_array, H_split_array=H_split_array, user_coordinates=user_coordinates)