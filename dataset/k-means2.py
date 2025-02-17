from mimo_rayleigh_lowv1 import *
import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def load_data_from_file(channel_path):
    data = np.load(channel_path)
    H_array = data['H_array']         # 形状 (600, 16, 40)
    H_split_array = data['H_split_array']  # 形状 (600, 2, 16, 40)
    user_coordinates = data['user_coordinates']  # 形状 (600, 40, 2)
    return H_array, H_split_array, user_coordinates

def calculate_correlation_matrix(H):
    """
    计算复数信道矩阵 H 的相关性矩阵
    H: 形状为 (num_antennas, num_users) 的复数信道矩阵
    返回: 形状为 (num_users, num_users) 的相关性矩阵（取模值）
    """
    # 计算每个用户信道向量的范数（注意沿天线维度，即 axis=0）
    norms = np.linalg.norm(H, axis=0, keepdims=True)
    # 防止除零，加上一个很小的数 1e-12
    H_normalized = H / (norms + 1e-12)
    # 利用共轭转置计算内积，得到用户间余弦相似度，并取绝对值
    correlation_matrix = np.abs(np.dot(H_normalized.T.conj(), H_normalized))
    return correlation_matrix

def cluster_users(correlation_matrix, num_clusters):
    """
    利用 SVD 对相关性矩阵降维，再使用 K-means 对用户进行聚类
    """
    U, S, Vt = np.linalg.svd(correlation_matrix)
    # 采用奇异值加权的前 num_clusters 个特征作为聚类输入
    features = U[:, :num_clusters] * S[:num_clusters]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(features)
    return kmeans.labels_

def process_timeslot(H, num_clusters):
    """
    对单个时隙的信道矩阵 H 计算相关性矩阵，并进行用户聚类
    """
    correlation_matrix = calculate_correlation_matrix(H)
    labels = cluster_users(correlation_matrix, num_clusters)
    return correlation_matrix, labels

# 设置中文字体支持
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

def plot_clusters(user_coordinates, labels, timeslot):
    """
    对给定时隙的用户坐标及聚类标签进行可视化
    user_coordinates: 形状 (num_users, 2)
    labels: 聚类结果标签数组
    timeslot: 当前时隙索引，用于图标题显示
    """
    x, y = user_coordinates[:, 0], user_coordinates[:, 1]
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=labels, cmap='viridis', marker='o',
                          edgecolor='w', linewidth=0.5, s=50, alpha=0.9)
    plt.title(f"用户聚类结果 - TTI {timeslot}", fontsize=14, fontweight='bold')
    plt.xlabel("X 坐标", fontsize=12)
    plt.ylabel("Y 坐标", fontsize=12)
    cbar = plt.colorbar(scatter)
    cbar.set_label('聚类标签', fontsize=12)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    channel_path = 'D:/python_project/ElegantRL-master/hellomimo/dataset/train/Rayleigh/uplink_40.npz'
    H_array, H_split_array, user_coordinates = load_data_from_file(channel_path)
    num_clusters = 3

    # 并行计算每个时隙的相关性矩阵和聚类结果
    results = Parallel(n_jobs=-1)(
        delayed(process_timeslot)(H_array[t], num_clusters) for t in range(H_array.shape[0])
    )

    # 解包结果
    correlation_matrices = [result[0] for result in results]
    cluster_labels = [result[1] for result in results]

    print("用户信道向量之间的相关性矩阵（每个 TTI）：")
    for i, matrix in enumerate(correlation_matrices):
        print(f"TTI {i}:")
        print(matrix)

    print("\n聚类结果（每个时隙）：")
    for i, labels in enumerate(cluster_labels):
        print(f"TTI {i}:")
        print(labels)

    # 可视化第一个时隙的聚类结果
    plot_clusters(user_coordinates[0], cluster_labels[0], 0)
