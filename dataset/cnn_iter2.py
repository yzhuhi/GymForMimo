import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

############################################
#             CNN 模型定义                 #
############################################
class CNN(nn.Module):
    def __init__(self, all_actions: int, actor_lr: float):
        super(CNN, self).__init__()
        # 使用 BCEWithLogitsLoss 来提高数值稳定性（输入为 logits）
        self.criterion = nn.BCEWithLogitsLoss()

        # CNN 层，加入 BatchNorm 缓解内部协变量偏移
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # MLP 层，用于处理 CNN 提取的特征，添加 Dropout 防止过拟合
        self.actor_mlp = nn.Sequential(
            nn.Linear(32 * 2 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Fair MLP 层，用于处理公平性指标输入（例如调度算法输出的用户速率信息）
        self.actor_fair_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(all_actions, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 最后一层：将两部分特征拼接后映射到各动作的 logits
        self.actor_last_layer = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, all_actions)
            # 输出为 logits，不接 Sigmoid，因为 BCEWithLogitsLoss 内部包含 Sigmoid
        )

        # 优化器：统一使用 self.parameters() 管理所有参数
        self.actor_optim = optim.Adam(self.parameters(), lr=actor_lr)

    def forward(self, x):
        # 假定输入字典中的 "channel" 和 "fair" 已经为 Tensor 且在正确设备上
        channel = x["channel"]
        fair = x["fair"]

        a1 = self.actor_cnn(channel)
        a2 = self.actor_mlp(a1)
        f2 = self.actor_fair_mlp(fair)
        combined_features = torch.cat((a2, f2), dim=1)
        logits = self.actor_last_layer(combined_features)
        return logits  # 返回 logits

    def get_binary_output(self, logits):
        # 推理时，先用 Sigmoid 转换为概率，再二值化
        probs = torch.sigmoid(logits)
        return (probs > 0.5).float()

    def compute_loss(self, logits, labels):
        return self.criterion(logits, labels)


############################################
#           调度算法（ZF 策略）            #
############################################
def proportional_fairness_algorithm_zf(H_array, n_actions=8, t_c=10, p_u=10):
    """
    基于比例公平的零迫ZF调度算法
    """
    batch_size, n_antennas, n_users = H_array.shape
    labels = np.zeros((batch_size, n_users), dtype=int)
    total_rates = []
    fairness_list = []
    rates_list = []
    R = np.zeros(n_users)
    user_rates = np.zeros(n_users)

    for b in range(batch_size):
        H = H_array[b]
        selected_users = set()
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
                    sinr_k = p_u / norm_a_k
                    r_i_t = np.log2(1 + sinr_k)
                    priority = r_i_t / (R[user] + 1e-8)
                    if priority > max_priority:
                        max_priority = priority
                        best_user = user
                if best_user != -1:
                    selected_users.add(best_user)
                    selected_H = np.hstack((selected_H, H[:, best_user].reshape(-1, 1)))
                    R[best_user] = (1 - 1 / t_c) * R[best_user] + (1 / t_c) * r_i_t
            for user in range(n_users):
                if user not in selected_users:
                    R[user] = (1 - 1 / t_c) * R[user]
        if selected_users:
            pseudo_inverse = np.linalg.pinv(selected_H.T @ selected_H) @ selected_H.T
            for idx, user in enumerate(selected_users):
                a_k_H = pseudo_inverse[idx]
                norm_a_k = np.linalg.norm(a_k_H) ** 2
                sinr_k = p_u / norm_a_k
                user_rates[user] = np.log2(1 + sinr_k)
        for user in selected_users:
            labels[b, user] = 1
        total_rate = np.sum(user_rates)
        fairness = (np.sum(user_rates) ** 2) / (n_actions * np.sum(user_rates ** 2) + 1e-8)
        total_rates.append(total_rate)
        fairness_list.append(fairness)
        rates_list.append(user_rates.copy())
    return labels, total_rates, fairness_list, rates_list


def greedy_scheduling_zf(H_array, n_actions=8, p_u=10):
    """
    基于零迫ZF的贪婪调度算法
    """
    batch_size, n_antennas, n_users = H_array.shape
    labels = np.zeros((batch_size, n_users), dtype=int)
    total_rates = []
    fairness_list = []
    rates_list = []
    user_rates = np.zeros(n_users)

    for b in range(batch_size):
        H = H_array[b]
        selected_users = set()
        for _ in range(n_actions):
            selected_H = H[:, list(selected_users)] if selected_users else np.zeros((n_antennas, 0))
            remaining_users = [i for i in range(n_users) if i not in selected_users]
            if selected_H.shape[1] < n_antennas and remaining_users:
                max_rate = -np.inf
                best_user = -1
                for user in remaining_users:
                    candidate_H = np.hstack((selected_H, H[:, user].reshape(-1, 1)))
                    if candidate_H.shape[1] > n_antennas:
                        continue
                    pseudo_inverse = np.linalg.pinv(candidate_H.T @ candidate_H) @ candidate_H.T
                    a_k_H = pseudo_inverse[-1]
                    norm_a_k = np.linalg.norm(a_k_H) ** 2
                    sinr_k = p_u / norm_a_k
                    r_i_t = np.log2(1 + sinr_k)
                    if r_i_t > max_rate:
                        max_rate = r_i_t
                        best_user = user
                if best_user != -1:
                    selected_users.add(best_user)
                    selected_H = np.hstack((selected_H, H[:, best_user].reshape(-1, 1)))
        if selected_users:
            pseudo_inverse = np.linalg.pinv(selected_H.T @ selected_H) @ selected_H.T
            for idx, user in enumerate(selected_users):
                a_k_H = pseudo_inverse[idx]
                norm_a_k = np.linalg.norm(a_k_H) ** 2
                sinr_k = p_u / norm_a_k
                user_rates[user] = np.log2(1 + sinr_k)
        for user in selected_users:
            labels[b, user] = 1
        total_rate = np.sum(user_rates)
        fairness = (total_rate ** 2) / (n_actions * np.sum(user_rates ** 2) + 1e-8)
        total_rates.append(total_rate)
        fairness_list.append(fairness)
        rates_list.append(user_rates.copy())
        user_rates[:] = 0.0
    return labels, total_rates, fairness_list, rates_list


############################################
#             数据加载函数                 #
############################################
def load_data_from_file(channel_path):
    data = np.load(channel_path)
    H_array = data['H_array']         # 例如形状 (600, 16, 40)
    H_split_array = data['H_split_array']  # 例如形状 (600, 2, 16, 40)
    user_coordinates = data['user_coordinates']  # 例如形状 (600, 40, 2)
    return H_array, H_split_array, user_coordinates


############################################
#          训练与推理主流程               #
############################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- 数据加载 --------------------
    channel_path = 'D:/python_project/GymFormimo/dataset/train/Rayleigh/uplink_30_1.npz'
    H_array, H_split_array, user_coordinates = load_data_from_file(channel_path)
    # 训练时取部分数据
    H_array = H_array[:12800]

    # -------------------- 训练超参数 --------------------
    batch_size = 64
    n_batches = len(H_array) // batch_size
    n_actions = 8
    all_actions = 16
    p_u = 10
    t_c = 50
    n_epochs = 100
    # 降低学习率，保证训练稳定（例如 1e-3）
    learning_rate = 1e-3

    # -------------------- 模型初始化 --------------------
    model = CNN(all_actions=all_actions, actor_lr=learning_rate).to(device)
    scheduler = optim.lr_scheduler.StepLR(model.actor_optim, step_size=50, gamma=0.1)

    # -------------------- 训练阶段 --------------------
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            H_batch = H_array[start_idx:end_idx]
            H_split_array_batch = H_split_array[start_idx:end_idx]

            # 生成标签：调用比例公平ZF算法
            label_batch, _, _, rates_list = proportional_fairness_algorithm_zf(
                H_batch, n_actions=n_actions, t_c=t_c, p_u=p_u
            )

            # 数据预处理：转换为 Tensor 并确保为 float 类型
            channel_input = torch.from_numpy(H_split_array_batch).float().to(device)
            # fair 输入尺寸需与模型输入保持一致，示例为 (batch_size, all_actions)
            if batch_idx != 0:
                fair_input = torch.from_numpy(np.array(rates_list)).float().to(device)
            else:
                fair_input = torch.zeros((batch_size, all_actions), device=device)
            input_data = {"channel": channel_input, "fair": fair_input}

            # 前向传播
            logits = model(input_data)
            labels_tensor = torch.from_numpy(label_batch).float().to(device)

            # 损失计算及反向传播
            loss = model.compute_loss((logits), labels_tensor)
            model.actor_optim.zero_grad()
            loss.backward()
            # 加入梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            model.actor_optim.step()

            epoch_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/n_batches:.4f}")

    # 保存训练好的模型（如有需要）
    # torch.save(model.state_dict(), 'cnn_actor_model.pth')

    # -------------------- 推理阶段 --------------------
    # 以下推理部分代码可按需要启用
    """
    model = CNN(all_actions=all_actions, actor_lr=1e-3).to(device)
    model.load_state_dict(torch.load('cnn_actor_model.pth'))
    model.eval()

    # 选择测试数据（例如取 32 个样本）
    H_array_test = H_array[12800:12832]
    print("Test H_array shape:", H_array_test.shape)

    channel_input_test = torch.from_numpy(H_split_array[12800:12832]).float().to(device)
    fair_input_test = torch.zeros((channel_input_test.shape[0], all_actions), device=device)
    input_data = {"channel": channel_input_test, "fair": fair_input_test}

    with torch.no_grad():
        logits = model(input_data)
        binary_output = model.get_binary_output(logits)

    print("Predicted logits:", logits)
    print("Binary output:", binary_output)
    """
