import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
import os
os.environ["OMP_NUM_THREADS"] = "1"


############################################
#      聚类相关函数（预分组模块）         #
############################################
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
    return np.stack(cluster_labels_batch, axis=0)


def pre_group_users(H_split_batch, cluster_labels_batch, n_clusters, fixed_users):
    """
    将原始的用户信道数据按聚类结果分组，每组固定选取 fixed_users 个用户
    H_split_batch: shape (batch_size, 2, n_antennas, n_users)
    cluster_labels_batch: shape (batch_size, n_users)
    返回：字典，每个 key 为 "cluster_i"，对应数据 shape (batch_size, 2, n_antennas, fixed_users)
    """
    batch_size = H_split_batch.shape[0]
    cluster_dict = {f"cluster_{i}": [] for i in range(n_clusters)}
    for b in range(batch_size):
        labels = cluster_labels_batch[b]  # shape (n_users,)
        for c in range(n_clusters):
            indices = np.where(labels == c)[0]
            if len(indices) >= fixed_users:
                selected_indices = indices[:fixed_users]
            else:
                # 若不足 fixed_users，则补 0（注意：这种简单补 0 的方式可能需要根据实际情况改进）
                selected_indices = np.concatenate([indices, np.zeros(fixed_users - len(indices), dtype=int)])
            # 选取对应的用户信道数据：H_split_batch[b] 形状 (2, n_antennas, n_users)
            cluster_data = H_split_batch[b, :, :, selected_indices]  # shape (2, n_antennas, fixed_users)
            cluster_dict[f"cluster_{c}"].append(cluster_data)
    # 转换为 np.array
    for c in range(n_clusters):
        cluster_dict[f"cluster_{c}"] = np.stack(cluster_dict[f"cluster_{c}"],
                                                axis=0)  # shape (batch_size, 2, n_antennas, fixed_users)
    return cluster_dict


############################################
#    基于预分组的多分支 CNN 模型定义       #
############################################
class ClusterPreGroupCNN(nn.Module):
    def __init__(self, n_clusters: int, fixed_users: int, branch_output_dim: int, final_output_dim: int,
                 actor_lr: float):
        """
        n_clusters: 聚类簇数（例如 3）
        fixed_users: 每个簇固定的用户数量
        branch_output_dim: 每个分支输出的特征维度
        final_output_dim: 最终输出维度（例如总的动作数，可理解为对各簇候选用户的二值选择结果）
        """
        super(ClusterPreGroupCNN, self).__init__()
        self.n_clusters = n_clusters
        self.fixed_users = fixed_users

        # 定义每个分支（可以考虑各分支权重共享或者独立，这里使用独立分支）
        # 每个分支采用类似于原始 CNN 模型中的卷积 + MLP 部分（这里假设输入尺寸为 (batch_size, 2, n_antennas, fixed_users)）
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
                nn.Linear(32 * 2 * (fixed_users // 4), branch_output_dim),  # 注意：假设 fixed_users 被池化后除以4
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.branches.append(branch)

        # 最后一层，将各分支特征拼接后输出最终结果
        self.final_fc = nn.Sequential(
            nn.Linear(n_clusters * branch_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, final_output_dim)  # 输出 logits（不经过 Sigmoid，损失函数内部处理）
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
            out = self.branches[i](branch_input)  # shape (batch_size, branch_output_dim)
            branch_outs.append(out)
        combined = torch.cat(branch_outs, dim=1)  # shape (batch_size, n_clusters * branch_output_dim)
        logits = self.final_fc(combined)  # shape (batch_size, final_output_dim)
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
    H_array = data['H_array']  # 例如形状 (600, 16, 40)
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
    # 为了示例，训练时取部分数据
    H_array_train = H_array[:12800]  # shape (12800, 16, 40)
    H_split_array_train = H_split_array[:12800]  # shape (12800, 2, 16, 40)

    # 设定聚类数与每簇固定用户数（例如：3簇，每簇固定选取 10 个用户）
    n_clusters = 2
    fixed_users = 8

    # -------------------- 预分组 --------------------
    # 对每个时隙（样本）基于 H_array_train 计算聚类标签
    cluster_labels_batch = compute_cluster_labels_batch(H_array_train, n_clusters)  # shape (batch_size, 40)
    # 根据聚类标签，将 H_split_array_train 分簇（注意：此处 H_split_array_train 为 numpy 数组）
    cluster_inputs = pre_group_users(H_split_array_train, cluster_labels_batch, n_clusters, fixed_users)
    # 现在 cluster_inputs 是一个字典，每个键对应形状 (batch_size, 2, 16, fixed_users)

    # -------------------- 模型初始化 --------------------
    # 假设每个分支输出特征维度设为 32，最终输出维度设为 3 * fixed_users（对每个簇进行二值选择）
    branch_output_dim = 32
    final_output_dim = n_clusters * fixed_users
    actor_lr = 1e-3
    model = ClusterPreGroupCNN(n_clusters, fixed_users, branch_output_dim, final_output_dim, actor_lr).to(device)

    # 示例：将预分组结果转换为 Tensor 并送入模型
    for key in cluster_inputs:
        # 转换为 tensor，并确保数据类型为 float32
        cluster_inputs[key] = torch.from_numpy(cluster_inputs[key]).float().to(device)

    # 假设标签为二值矩阵，形状与 final_output_dim 相同（这里仅构造随机标签作示例）
    batch_size = cluster_inputs["cluster_0"].shape[0]
    labels_np = np.random.randint(0, 2, size=(batch_size, final_output_dim)).astype(np.float32)
    labels_tensor = torch.from_numpy(labels_np).to(device)

    # 前向传播、计算损失、反向传播（示例一个 batch 的流程）
    logits = model(cluster_inputs)
    loss = model.compute_loss(logits, labels_tensor)
    model.actor_optim.zero_grad()
    loss.backward()
    model.actor_optim.step()

    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())

    # 推理示例：得到二值输出
    with torch.no_grad():
        binary_output = model.get_binary_output(logits)
    print("Binary output shape:", binary_output.shape)
