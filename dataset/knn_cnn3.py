import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")

from sklearn.cluster import KMeans
import os, gc

os.environ["OMP_NUM_THREADS"] = "1"

############################################
#         比例公平（PF）调度算法         #
############################################
def proportional_fairness_algorithm_zf(H_array, n_actions=8, t_c=10, p_u=10):
    """
    PF 调度算法，根据原始信道 H_array 计算用户选择标签
    输入:
      H_array: shape (N, n_antennas, n_users)
    输出:
      labels: 二值矩阵，形状 (N, n_users)，1 表示该用户被选中
      total_rates, fairness_list, rates_list: 其它指标（此处仅返回 labels）
    """
    N, n_antennas, n_users = H_array.shape
    labels = np.zeros((N, n_users), dtype=int)
    total_rates = []
    fairness_list = []
    rates_list = []
    for b in range(N):
        H = H_array[b]
        selected_users = set()
        R = np.zeros(n_users)
        user_rates = np.zeros(n_users)
        for _ in range(n_actions):
            selected_H = H[:, list(selected_users)] if selected_users else np.zeros((n_antennas, 0))
            remaining_users = [i for i in range(n_users) if i not in selected_users]
            if selected_H.shape[1] < n_antennas and remaining_users:
                max_priority = -np.inf
                best_user = -1
                for user in remaining_users:
                    candidate_H = np.hstack((selected_H, H[:, user].reshape(-1, 1)))
                    if candidate_H.shape[1] > n_antennas:
                        continue
                    pseudo_inverse = np.linalg.pinv(candidate_H.T @ candidate_H) @ candidate_H.T
                    a_k_H = pseudo_inverse[-1]
                    norm_a_k = np.linalg.norm(a_k_H) ** 2
                    sinr_k = p_u / (norm_a_k + 1e-8)
                    r_i_t = np.log2(1 + sinr_k)
                    priority = r_i_t / (R[user] + 1e-8)
                    if priority > max_priority:
                        max_priority = priority
                        best_user = user
                if best_user != -1:
                    selected_users.add(best_user)
                    selected_H = np.hstack((selected_H, H[:, best_user].reshape(-1, 1)))
                    R[best_user] = (1 - 1/t_c) * R[best_user] + (1/t_c) * r_i_t
            for user in range(n_users):
                if user not in selected_users:
                    R[user] = (1 - 1/t_c) * R[user]
        if selected_users:
            pseudo_inverse = np.linalg.pinv(selected_H.T @ selected_H) @ selected_H.T
            for idx, user in enumerate(selected_users):
                a_k_H = pseudo_inverse[idx]
                norm_a_k = np.linalg.norm(a_k_H) ** 2
                sinr_k = p_u / (norm_a_k + 1e-8)
                user_rates[user] = np.log2(1 + sinr_k)
        for user in selected_users:
            labels[b, user] = 1
        total_rate = np.sum(user_rates)
        fairness = (np.sum(user_rates)**2) / (n_actions * np.sum(user_rates**2) + 1e-8)
        total_rates.append(total_rate)
        fairness_list.append(fairness)
        rates_list.append(user_rates.copy())
    return labels, total_rates, fairness_list, rates_list

############################################
#      聚类相关函数（预分组模块）         #
############################################
def calculate_correlation_matrix(H):
    """
    H: shape (n_antennas, n_users)
    返回 shape (n_users, n_users) 的相关性矩阵
    """
    norms = np.linalg.norm(H, axis=0, keepdims=True)
    H_normalized = H / (norms + 1e-12)
    correlation_matrix = np.abs(np.dot(H_normalized.T.conj(), H_normalized))
    return correlation_matrix

def cluster_users(correlation_matrix, num_clusters):
    U, S, _ = np.linalg.svd(correlation_matrix)
    features = U[:, :num_clusters] * S[:num_clusters]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(features)
    return kmeans.labels_

def compute_cluster_labels_batch(H_array_batch, n_clusters):
    cluster_labels_batch = []
    for H in H_array_batch:
        corr_mat = calculate_correlation_matrix(H)
        labels = cluster_users(corr_mat, n_clusters)
        cluster_labels_batch.append(labels)
        gc.collect()
    return np.stack(cluster_labels_batch, axis=0)

def balance_clusters(cluster_labels, n_clusters, fixed_users):
    """
    对单个样本的聚类标签进行平衡分配，确保每个簇固定获得 fixed_users 个用户索引
    """
    n_users = len(cluster_labels)
    clusters = {i: list(np.where(cluster_labels == i)[0]) for i in range(n_clusters)}
    all_users = set(range(n_users))
    balanced = {}
    for c in range(n_clusters):
        indices = clusters[c]
        if len(indices) >= fixed_users:
            balanced[c] = np.random.choice(indices, fixed_users, replace=False).tolist()
        else:
            balanced[c] = indices.copy()
            deficit = fixed_users - len(indices)
            candidates = list(all_users - set(indices))
            if len(candidates) < deficit:
                fill = list(np.random.choice(candidates, deficit, replace=True))
            else:
                fill = list(np.random.choice(candidates, deficit, replace=False))
            balanced[c].extend(fill)
    return balanced

def pre_group_users(H_split_batch, cluster_labels_batch, n_clusters, fixed_users):
    """
    H_split_batch: shape (batch_size, 2, n_antennas, n_users)
    cluster_labels_batch: shape (batch_size, n_users)
    返回两个字典：
       cluster_dict: { "cluster_i": shape (batch_size, 2, n_antennas, fixed_users) }
       cluster_indices: { "cluster_i": shape (batch_size, fixed_users) }，记录候选用户索引
    """
    batch_size = H_split_batch.shape[0]
    cluster_dict = {f"cluster_{i}": [] for i in range(n_clusters)}
    cluster_indices = {f"cluster_{i}": [] for i in range(n_clusters)}
    for b in range(batch_size):
        labels = cluster_labels_batch[b]  # shape (n_users,)
        balanced = balance_clusters(labels, n_clusters, fixed_users)
        for c in range(n_clusters):
            selected_indices = np.array(balanced[c])
            cluster_indices[f"cluster_{c}"].append(selected_indices)
            # 使用 np.take 在用户轴（轴3）选取，保证输出形状 (2, n_antennas, fixed_users)
            cluster_data = H_split_batch[b, :, :, selected_indices]
            cluster_data = np.transpose(cluster_data, (1, 0, 2))

            cluster_dict[f"cluster_{c}"].append(cluster_data)
    for c in range(n_clusters):
        cluster_dict[f"cluster_{c}"] = np.stack(cluster_dict[f"cluster_{c}"], axis=0)
        cluster_indices[f"cluster_{c}"] = np.stack(cluster_indices[f"cluster_{c}"], axis=0)
    return cluster_dict, cluster_indices

############################################
#    基于预分组的多分支 CNN 模型定义       #
############################################
class ClusterPreGroupCNN(nn.Module):
    def __init__(self, n_clusters: int, fixed_users: int, branch_output_dim: int, final_output_dim: int,
                 actor_lr: float):
        """
        n_clusters: 聚类簇数
        fixed_users: 每个簇固定的用户数量（要求 n_clusters * fixed_users == n_users）
        branch_output_dim: 每个分支输出的特征维度
        final_output_dim: 最终输出维度（应等于 n_users）
        """
        super(ClusterPreGroupCNN, self).__init__()
        self.n_clusters = n_clusters
        self.fixed_users = fixed_users

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
                # 假设 n_antennas=8, fixed_users=20（当 n_clusters=2，n_users=40），经过两次2×2池化后空间尺寸变为 (8/4, 20/4) = (2,5)
                # 平坦后尺寸 = 32 * 2 * 5 = 320
                nn.Linear(32 * (8 // 4) * (fixed_users // 4), branch_output_dim),
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
    H_array = data['H_array']         # 例如 (600, 16, 40)
    H_split_array = data['H_split_array']  # 例如 (600, 2, 16, 40)
    user_coordinates = data['user_coordinates']  # 例如 (600, 40, 2)
    return H_array, H_split_array, user_coordinates

############################################
#         自定义 Dataset 与 DataLoader         #
############################################
class PreGroupDataset(data.Dataset):
    def __init__(self, cluster_inputs, labels):
        """
        cluster_inputs: dict, 每个键对应一个 Tensor，形状 (N, 2, n_antennas, fixed_users)
        labels: Tensor, 形状 (N, n_users)
        """
        self.cluster_inputs = cluster_inputs
        self.labels = labels
        self.N = next(iter(cluster_inputs.values())).shape[0]
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        sample = {key: self.cluster_inputs[key][idx] for key in self.cluster_inputs}
        label = self.labels[idx]
        return sample, label

def collate_fn(batch):
    # batch 是 list of (sample, label)
    samples, labels = zip(*batch)
    collated = {}
    for key in samples[0]:
        collated[key] = torch.stack([s[key] for s in samples], dim=0)
    labels = torch.stack(labels, dim=0)
    return collated, labels

############################################
#          训练与测试主流程（示例）         #
############################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- 数据加载 --------------------
    channel_path = 'D:/python_project/GymFormimo/dataset/train/Rayleigh/uplink_30_1.npz'
    H_array, H_split_array, user_coordinates = load_data_from_file(channel_path)
    # 假设取 3200 个样本进行训练，另外一部分做测试
    H_array_train = H_array[:6400]            # (3200, 16, 40)
    H_split_array_train = H_split_array[:6400]  # (3200, 2, 16, 40)
    H_array_test = H_array[6400:7200]           # (800, 16, 40)
    H_split_array_test = H_split_array[6400:7200]  # (800, 2, 16, 40)

    # -------------------- PF 标签生成 --------------------
    pf_labels_train, _, _, _ = proportional_fairness_algorithm_zf(H_array_train, n_actions=8, t_c=10, p_u=10)
    # pf_labels_train: (3200, 40)
    pf_labels_test, _, _, _ = proportional_fairness_algorithm_zf(H_array_test, n_actions=8, t_c=10, p_u=10)

    # -------------------- 预分组 --------------------
    n_clusters = 2
    n_users = H_array_train.shape[2]  # 40
    fixed_users = n_users // n_clusters  # 20
    final_output_dim = n_clusters * fixed_users  # 40

    cluster_labels_batch_train = compute_cluster_labels_batch(H_array_train, n_clusters)  # (3200, 40)
    cluster_inputs_train, cluster_indices_train = pre_group_users(H_split_array_train, cluster_labels_batch_train, n_clusters, fixed_users)
    # cluster_inputs_train: dict, each (3200, 2, 16, 20)

    # 将 PF 标签映射到每个簇，构造目标标签（训练）
    grouped_labels_train = np.zeros((cluster_indices_train["cluster_0"].shape[0], n_clusters, fixed_users), dtype=np.float32)
    for b in range(grouped_labels_train.shape[0]):
        for c in range(n_clusters):
            candidate_idx = cluster_indices_train[f"cluster_{c}"][b]  # 长度 fixed_users
            grouped_labels_train[b, c, :] = pf_labels_train[b, candidate_idx]
    grouped_labels_train = grouped_labels_train.reshape(grouped_labels_train.shape[0], -1)  # (3200, 40)

    # 同理，测试数据预分组与标签映射
    cluster_labels_batch_test = compute_cluster_labels_batch(H_array_test, n_clusters)  # (800, 40)
    cluster_inputs_test, cluster_indices_test = pre_group_users(H_split_array_test, cluster_labels_batch_test, n_clusters, fixed_users)
    grouped_labels_test = np.zeros((cluster_indices_test["cluster_0"].shape[0], n_clusters, fixed_users), dtype=np.float32)
    for b in range(grouped_labels_test.shape[0]):
        for c in range(n_clusters):
            candidate_idx = cluster_indices_test[f"cluster_{c}"][b]
            grouped_labels_test[b, c, :] = pf_labels_test[b, candidate_idx]
    grouped_labels_test = grouped_labels_test.reshape(grouped_labels_test.shape[0], -1)  # (800, 40)

    # -------------------- 转换为 Tensor --------------------
    # 将预分组数据（字典中的每个项）转换为 Tensor（目前是 numpy 数组）
    for key in cluster_inputs_train:
        cluster_inputs_train[key] = torch.from_numpy(cluster_inputs_train[key]).float()
    labels_tensor_train = torch.from_numpy(grouped_labels_train).float()

    for key in cluster_inputs_test:
        cluster_inputs_test[key] = torch.from_numpy(cluster_inputs_test[key]).float()
    labels_tensor_test = torch.from_numpy(grouped_labels_test).float()

    # -------------------- 创建 Dataset 和 DataLoader --------------------
    train_dataset = PreGroupDataset(cluster_inputs_train, labels_tensor_train)
    test_dataset = PreGroupDataset(cluster_inputs_test, labels_tensor_test)
    batch_size = 64
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # -------------------- 模型初始化 --------------------
    branch_output_dim = 32
    actor_lr = 1e-3
    model = ClusterPreGroupCNN(n_clusters, fixed_users, branch_output_dim, final_output_dim, actor_lr).to(device)

    # -------------------- 训练循环 --------------------
    epochs = 300
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
            batch_labels = batch_labels.to(device)
            logits = model(batch_inputs)
            loss = model.compute_loss(logits, batch_labels)
            model.actor_optim.zero_grad()
            loss.backward()
            model.actor_optim.step()
            epoch_loss += loss.item() * batch_labels.size(0)
        epoch_loss /= len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "cluster_pre_group_cnn.pth")
    print("Model saved.")

    # -------------------- 测试阶段 --------------------
    model.eval()
    all_pf = []
    all_network = []
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
            batch_labels = batch_labels.to(device)
            logits_test = model(batch_inputs)
            binary_output = model.get_binary_output(logits_test)
            all_network.append(binary_output.cpu().numpy())
            all_pf.append(batch_labels.cpu().numpy())
    all_network = np.concatenate(all_network, axis=0)  # shape (N_test, n_users)
    all_pf = np.concatenate(all_pf, axis=0)            # shape (N_test, n_users)

    # 打印测试集样本0的整体选择结果对比
    print("样本0 PF 标签：", all_pf[0])
    print("样本0 网络选择结果：", all_network[0])
