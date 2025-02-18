from mimo_rayleigh_lowv1 import *

# H_array, H_split_array, user_coordinates: (600,16,40) (600,2,16,40) (600,40,2) 600:时隙 16:基站天线数量 40:单天线用户数
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def load_data_from_file(channel_path):
    data = np.load(channel_path)
    H_array = data['H_array']
    H_split_array = data['H_split_array']
    user_coordinates = data['user_coordinates']
    return H_array, H_split_array, user_coordinates

def calculate_correlation_matrix(H):
    num_users = H.shape[1]
    correlation_matrix = np.zeros((num_users, num_users), dtype=np.complex128)
    for i in range(num_users):
        for j in range(num_users):
            correlation_matrix[i, j] = np.vdot(H[:, i], H[:, j]) / (np.linalg.norm(H[:, i]) * np.linalg.norm(H[:, j]))
    return np.abs(correlation_matrix)  # 返回模值


def cluster_users(correlation_matrix, num_clusters):
    U, S, Vt = np.linalg.svd(correlation_matrix)
    features = U[:, :num_clusters]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    return kmeans.labels_

def process_timeslot(H, num_clusters):
    correlation_matrix = calculate_correlation_matrix(H)
    labels = cluster_users(correlation_matrix, num_clusters)
    return correlation_matrix, labels


from matplotlib import rcParams
from matplotlib import font_manager

# 设置中文字体为 SimHei（黑体）或其他支持中文的字体
rcParams['font.sans-serif'] = ['SimHei']  # SimHei 是常见的中文字体，确保系统已安装
rcParams['axes.unicode_minus'] = False  # 避免负号显示问题

def plot_clusters(user_coordinates, labels, timeslot):
    x, y = user_coordinates[:, 0], user_coordinates[:, 1]
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=labels, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.8)
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
    num_clusters = 2

    # 并行计算每个时隙的相关性矩阵和聚类结果
    results = Parallel(n_jobs=-1)(
        delayed(process_timeslot)(H_array[t], num_clusters) for t in range(H_array.shape[0])
    )

    # 解包 results 中的相关性矩阵和聚类标签
    correlation_matrices = [result[0] for result in results]
    cluster_labels = [result[1] for result in results]

    print("用户信道向量之间的相关性矩阵（每个TTI）：")
    for i, matrix in enumerate(correlation_matrices):
        print(f"TTI {i}:")
        print(matrix)

    print("\n聚类结果（每个时隙）：")
    for i, labels in enumerate(cluster_labels):
        print(f"TTI {i}:")
        print(labels)

    # 可视化第一个时隙的结果
    plot_clusters(user_coordinates[0], cluster_labels[0], 0)




