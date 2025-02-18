import copy
import os
import sys
import numpy as np

# 获取当前文件及其父路径
try:
    curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
    parent_path = os.path.dirname(curr_path)  # 父路径
    sys.path.append(parent_path)  # 添加路径到系统路径
except Exception as e:
    print(f"路径处理错误: {e}")



# 相关系数矩阵Rtx,Rrx
def correlationmatrix(nr, nt, pt, pr):
    Rtx = np.zeros((nt, nt))
    Rrx = np.zeros((nr, nr))

    # Generate transmit correlation matrix
    for i in range(nt):
        Rtx[i] = pt ** np.abs(np.arange(nt) - i)

    # Generate receive correlation matrix
    for i in range(nr):
        Rrx[i] = pr ** np.abs(np.arange(nr) - i)

    return Rtx, Rrx

# 独立衰落信道
def generate_channel(time_step, nt, users, nr, user_coordinate,d0=1, pl_gamma=3.5, shadowing_sigma=6, downlink=True):
    try:
        R = 100  # 基站覆盖半径
        d_users_origin = np.zeros(users)
        A = 62000  # 转换为线性值

        # 用户位置随机初始化
        if time_step == 0:
            user_coordinates = []
            for _ in range(users):
                sigma = 2 * np.pi * np.random.rand()
                r = np.sqrt(np.random.rand() * (R ** 2 - d0 ** 2) + d0 ** 2)
                x = r * np.cos(sigma)
                y = r * np.sin(sigma)
                user_coordinates.append([x, y])
        else:
            user_coordinates = user_coordinate.copy()
            for i_user in range(users):
                sigma = 2 * np.pi * np.random.rand()
                r = 0 if np.random.random() < 0.3 else 0.01 * np.sqrt(np.random.rand() * (R ** 2 - d0 ** 2) + d0 ** 2)
                user_coordinates[i_user][0] += r * np.cos(sigma)
                user_coordinates[i_user][1] += r * np.sin(sigma)
                dist_BaseToUser = np.sqrt(user_coordinates[i_user][0] ** 2 + user_coordinates[i_user][1] ** 2)
                # 确保用户位置不超过边界
                if dist_BaseToUser > R:
                    user_coordinates[i_user][0] -= 2 * (dist_BaseToUser - R) * np.cos(sigma)
                    user_coordinates[i_user][1] -= 2 * (dist_BaseToUser - R) * np.sin(sigma)
                elif dist_BaseToUser < d0:
                    user_coordinates[i_user][0] -= 2 * (d0 - dist_BaseToUser) * np.cos(sigma)
                    user_coordinates[i_user][1] -= 2 * (d0 - dist_BaseToUser) * np.sin(sigma)

        # 求用户到基站距离以及信道矩阵
        if downlink:
            H = np.zeros((users, nt), dtype=np.complex128)
            for i_us in range(users):
                u_i = np.random.randn(1, nt) + 1j * np.random.randn(1, nt)
                d_users_origin[i_us] = np.linalg.norm(user_coordinates[i_us])
                pl = d_users_origin[i_us] ** (-pl_gamma)
                shadowing = shadowing_sigma * np.random.randn()
                beta = A * np.power(10, shadowing / 10) * pl
                R_i_sqrt = np.sqrt(beta) * np.eye(nt)
                H[i_us, :] = u_i @ R_i_sqrt
        else:
            H = np.zeros((nt, users), dtype=np.complex128)
            for i_us in range(users):
                u_i = np.random.randn(nt) + 1j * np.random.randn(nt)
                d_users_origin[i_us] = np.linalg.norm(user_coordinates[i_us])
                pl = d_users_origin[i_us] ** (-pl_gamma)
                shadowing = shadowing_sigma * np.random.randn()
                beta = A * np.power(10, shadowing / 10) * pl
                R_i_sqrt = np.sqrt(beta) * np.eye(nt)
                H[:, i_us] = R_i_sqrt @ u_i

        return H, d_users_origin, user_coordinates

    except Exception as e:
        print(f"发生错误: {e}")
        return None, None, None

# 相关衰落瑞利信道
def generate_data(timesteps, nt, users, nr, pt, pr, random_seed=34, downlink=False):
    try:
        users_coord = []
        users_cor = []
        H_array_shape = (timesteps, users, nt) if downlink else (timesteps, nt, users)
        H_array = np.zeros(H_array_shape, dtype=np.complex128)
        H_split_array = np.zeros((timesteps, 2, *H_array.shape[1:]))

        for n_obs in range(timesteps):
            np.random.seed(random_seed + n_obs)  # 设置局部随机数种子

            H, _, user_coordinates = generate_channel(n_obs, nt, users, nr, users_cor, d0=1, pl_gamma=3.5, shadowing_sigma=6, downlink=downlink)
            users_cor = copy.deepcopy(user_coordinates)
            if user_coordinates is None:
                raise ValueError("用户坐标生成失败！")

            users_coord.append(user_coordinates)
            H_split = np.zeros((2, H.shape[0], H.shape[1]))

            # 根据 downlink 的情况调整 H 乘上相关矩阵
            if downlink:
                Rtx, Rrx = correlationmatrix(users, nt, pt, pr)
                H = H @ Rtx  # 对信道矩阵乘上发送相关矩阵
            else:
                Rtx, Rrx = correlationmatrix(nt, users, pt, pr)
                H = Rrx @ H  # 对信道矩阵乘上接收相关矩阵

            H_split[0, :, :] = np.real(H)
            H_split[1, :, :] = np.imag(H)
            H_array[n_obs] = H
            H_split_array[n_obs] = H_split

        return H_array, H_split_array, users_coord

    except Exception as e:
        print(f"发生错误: {e}")
        return None, None, None


def generate_and_save_data():
    # 设置参数
    timesteps = 64000  # 信道矩阵个数
    nt = 8  # 基站天线数
    users = 16  # 用户总数
    nr = 1  # 单用户天线数
    pt = 0.5  # 相关因子
    pr = 0.5
    downlink = False

    # 生成数据
    H_array, H_split_array, user_coordinates = generate_data(timesteps, nt, users, nr, pt, pr, downlink=downlink)

    # 保存数据到文件
    save_data_to_file(H_array, H_split_array, user_coordinates)


def save_data_to_file(H_array, H_split_array, user_coordinates):
    curr_path = os.getcwd()  # 获取当前路径

    test_path = os.path.join(curr_path, 'test', 'Rayleigh', 'uplink_30_1.npz')

    # 创建目录
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    # 保存为npz格式
    np.savez(test_path, H_array=H_array, H_split_array=H_split_array, user_coordinates=user_coordinates)


# def load_data_from_file(file_path):
#     data = np.load(file_path)
#
#     # 从npz文件中获取数据
#     H_array = data['H_array']
#     H_split_array = data['H_split_array']
#     user_coordinates = data['user_coordinates']
#
#     return H_array, H_split_array, user_coordinates



if __name__ == '__main__':
    generate_and_save_data()



