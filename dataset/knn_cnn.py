import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
import os, gc
os.environ["OMP_NUM_THREADS"] = "1"

############################################
#      聚类相关函数（预分组模块）         #
############################################
def calculate_correlation_matrix(H):
    """
    计算复数信道矩阵 H 的相关性矩阵
    H: 形状为 (num_antennas, n_users) 的复数信道矩阵
    返回: 形状为 (n_users, n_users) 的相关性矩阵（取模值）
    """
    norms = np.linalg.norm(H, axis=0, keepdims=True)
    H_normalized = H / (norms + 1e-12)
    correlation_matrix = np.abs(np.dot(H_normalized.T.conj(), H_normalized))
    return correlation_matrix

def cluster_users(correlation_matrix, num_clusters):
    """
    利用 SVD 对相关性矩阵降维，再使用 K-means 对用户进行聚类
    """
    U, S, _ = np.linalg.svd(correlation_matrix)
    features = U[:, :num_clusters] * S[:num_clusters]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(features)
    return kmeans.labels_

def compute_cluster_labels_batch(H_array_batch, n_clusters):
    """
    对一批信道矩阵（形状 (batch_size, n_antennas, n_users)）计算聚类标签
    返回：shape (batch_size, n_users)
    """
    cluster_labels_batch = []
    for H in H_array_batch:
        corr_mat = calculate_correlation_matrix(H)
        labels = cluster_users(corr_mat, n_clusters)
        cluster_labels_batch.append(labels)
        gc.collect()
    return np.stack(cluster_labels_batch, axis=0)

def balance_clusters(cluster_labels, n_clusters, fixed_users):
    """
    对单个样本的聚类标签进行平衡分配，确保每个簇获得固定数量 fixed_users 的用户。
    cluster_labels: 形状 (n_users,) 的数组，表示每个用户的聚类标签
    返回：字典，键为簇号，值为长度为 fixed_users 的用户索引列表（可能包含重复）。
    """
    n_users = len(cluster_labels)
    clusters = {i: list(np.where(cluster_labels == i)[0]) for i in range(n_clusters)}
    all_users = set(range(n_users))
    balanced = {}
    for c in range(n_clusters):
        indices = clusters[c]
        if len(indices) >= fixed_users:
            # 随机选择 fixed_users 个
            balanced[c] = np.random.choice(indices, fixed_users, replace=False).tolist()
        else:
            # 不足时，取已有的并从未被选中的用户中随机补充
            balanced[c] = indices.copy()
            deficit = fixed_users - len(indices)
            # 可选候选集：所有用户中去掉已经属于该簇的
            candidates = list(all_users - set(indices))
            if len(candidates) < deficit:
                fill = list(np.random.choice(candidates, deficit, replace=True))
            else:
                fill = list(np.random.choice(candidates, deficit, replace=False))
            balanced[c].extend(fill)
    return balanced

def pre_group_users(H_split_batch, cluster_labels_batch, n_clusters, fixed_users):
    """
    将原始的用户信道数据按聚类结果分组，每组固定选取 fixed_users 个用户（采用重新分配策略）。
    H_split_batch: shape (batch_size, 2, n_antennas, n_users)
    cluster_labels_batch: shape (batch_size, n_users)
    返回：字典，每个 key 为 "cluster_i"，对应数据 shape (batch_size, 2, n_antennas, fixed_users)
    """
    batch_size = H_split_batch.shape[0]
    cluster_dict = {f"cluster_{i}": [] for i in range(n_clusters)}
    for b in range(batch_size):
        labels = cluster_labels_batch[b]  # shape (n_users,)
        # 平衡分配：确保每个簇固定获得 fixed_users 个用户索引
        balanced = balance_clusters(labels, n_clusters, fixed_users)
        for c in range(n_clusters):
            selected_indices = np.array(balanced[c])
            # 选取对应的用户信道数据
            cluster_data = H_split_batch[b, :, :, selected_indices]  # shape (2, n_antennas, fixed_users)
            cluster_dict[f"cluster_{c}"].append(cluster_data)
    for c in range(n_clusters):
        cluster_dict[f"cluster_{c}"] = np.stack(cluster_dict[f"cluster_{c}"], axis=0)
    return cluster_dict

############################################
#    基于预分组的多分支 CNN 模型定义       #
############################################
class ClusterPreGroupCNN(nn.Module):
    def __init__(self, n_clusters: int, fixed_users: int, branch_output_dim: int, final_output_dim: int,
                 actor_lr: float):
        """
        n_clusters: 聚类簇数（例如 2 或 3）
        fixed_users: 每个簇固定的用户数量
        branch_output_dim: 每个分支输出的特征维度
        final_output_dim: 最终输出维度（例如对各簇候选用户的二值选择结果总数）
        """
        super(ClusterPreGroupCNN, self).__init__()
        self.n_clusters = n_clusters
        self.fixed_users = fixed_users

        # 定义每个分支（各分支独立）
        self.branches = nn.ModuleList()
        for _ in range(n_clusters):
            branch = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                # 假设 n_antennas 固定为 16，经两次池化后尺寸为 (16/4=4, fixed_users/4)
                nn.Linear(32 * 4 * (fixed_users // 4), branch_output_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.branches.append(branch)

        self.final_fc = nn.Sequential(
            nn.Linear(n_clusters * branch_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, final_output_dim)
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.actor_optim = optim.Adam(self.parameters(), lr=actor_lr)

    def forward(self, x):
        """
        x: 字典，包含键 "cluster_0", "cluster_1", ... "cluster_{n_clusters-1}"
           每个值为 tensor，形状 (batch_size, 2, n_antennas, fixed_users)
        """
        branch_outs = []
        for i in range(self.n_clusters):
            branch_input = x[f"cluster_{i}"]
            out = self.branches[i](branch_input)
            branch_outs.append(out)
        combined = torch.cat(branch_outs, dim=1)
        logits = self.final_fc(combined)
        return logits

    def get_binary_output(self, logits):
        probs = torch.sigmoid(logits)
        return (probs > 0.5).float()

    def compute_loss(self, logits, labels):
        return self.criterion(logits, labels)

############################################
#          数据加载函数（示例）           #
############################################
def load_data_from_file(channel_path):
    data = np.load(channel_path)
    H_array = data['H_array']         # 例如形状 (600, 16, 40)
    H_split_array = data['H_split_array']  # 例如形状 (600, 2, 16, 40)
    user_coordinates = data['user_coordinates']  # 例如形状 (600, 40, 2)
    return H_array, H_split_array, user_coordinates

############################################
#          训练与推理主流程（示例）         #
############################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- 数据加载 --------------------
    channel_path = 'D:/python_project/GymFormimo/dataset/train/Rayleigh/uplink_30_1.npz'
    H_array, H_split_array, user_coordinates = load_data_from_file(channel_path)
    H_array_train = H_array[:6400]            # shape (6400, 16, 40)
    H_split_array_train = H_split_array[:6400]  # shape (6400, 2, 16, 40)

    # 设定聚类数与每簇固定用户数（例如：2簇，每簇固定选取 8 个用户）
    n_clusters = 2
    fixed_users = 8

    # -------------------- 预分组 --------------------
    cluster_labels_batch = compute_cluster_labels_batch(H_array_train, n_clusters)  # shape (batch_size, 40)
    cluster_inputs = pre_group_users(H_split_array_train, cluster_labels_batch, n_clusters, fixed_users)
    # cluster_inputs 为字典，每个键对应形状 (batch_size, 2, 16, fixed_users)

    # -------------------- 模型初始化 --------------------
    branch_output_dim = 32
    final_output_dim = n_clusters * fixed_users
    actor_lr = 1e-3
    model = ClusterPreGroupCNN(n_clusters, fixed_users, branch_output_dim, final_output_dim, actor_lr).to(device)

    # 将预分组结果转换为 Tensor 并送入模型
    for key in cluster_inputs:
        cluster_inputs[key] = torch.from_numpy(cluster_inputs[key]).float().to(device)

    # 构造随机标签（示例），形状 (batch_size, final_output_dim)
    batch_size = cluster_inputs["cluster_0"].shape[0]
    labels_np = np.random.randint(0, 2, size=(batch_size, final_output_dim)).astype(np.float32)
    labels_tensor = torch.from_numpy(labels_np).to(device)

    # 前向传播、计算损失、反向传播
    logits = model(cluster_inputs)
    loss = model.compute_loss(logits, labels_tensor)
    model.actor_optim.zero_grad()
    loss.backward()
    model.actor_optim.step()

    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())

    # 推理示例
    with torch.no_grad():
        binary_output = model.get_binary_output(logits)
    print("Binary output shape:", binary_output.shape)
