import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, all_actions: int, device: torch.device, actor_lr: float):
        super().__init__()
        self.criterion = nn.BCELoss()  # 使用BCE损失函数，适合二分类标签
        self.device = device

        # CNN层
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten()
        ).to(self.device)

        # MLP层
        self.actor_mlp = nn.Sequential(
            nn.Linear(32 * 2 * 4, 64),
            nn.ReLU()
        ).to(self.device)

        # Fair MLP层
        self.actor_fair_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(all_actions, 32),
            nn.ReLU()
        ).to(self.device)

        # 最后一层
        self.actor_last_layer = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, all_actions),
            nn.Sigmoid()
        ).to(self.device)

        # 优化器
        self.actor_optim = optim.Adam(
            list(self.actor_cnn.parameters()) +
            list(self.actor_mlp.parameters()) +
            list(self.actor_fair_mlp.parameters()) +
            list(self.actor_last_layer.parameters()),
            lr=actor_lr
        )

    def forward(self, x):
        # 返回连续概率
        channel = x["channel"] if torch.is_tensor(x["channel"]) else torch.Tensor(x["channel"])
        fair = x["fair"] if torch.is_tensor(x["fair"]) else torch.Tensor(x["fair"])

        channel = channel.to(self.device)
        fair = fair.to(self.device)

        a1 = self.actor_cnn(channel)
        a2 = self.actor_mlp(a1)
        f2 = self.actor_fair_mlp(fair)
        combined_actor_features = torch.cat((a2, f2), dim=1)
        action_logits_vec = self.actor_last_layer(combined_actor_features)
        return action_logits_vec  # 连续值，用于计算loss

    def get_binary_output(self, action_logits_vec):
        # 推理时调用，将连续值转化为二值
        return (action_logits_vec > 0.5).float()

    def compute_loss(self, predictions, labels):
        # BCELoss 用于标签为0/1的二分类
        return self.criterion(predictions, labels)

def proportional_fairness_algorithm_zf(H_array, n_actions=8, t_c=10, p_u=10):
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
                    R[best_user] = (1 - 1/t_c) * R[best_user] + (1/t_c) * r_i_t

            for user in range(n_users):
                if user not in selected_users:
                    R[user] = (1 - 1/t_c) * R[user]

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
    基于零迫ZF的贪婪调度算法：
    每轮都选择能带来最大瞬时速率的用户，直到达到指定的n_actions数量或空间不足以再增加用户。
    
    :param H_array: 信道矩阵, shape=[batch_size, n_antennas, n_users]
    :param n_actions: 贪婪调度每次选择的用户总数
    :param p_u: 用户发射功率
    :return:
        labels: shape=[batch_size, n_users], 每个batch被选择用户标记为1
        total_rates: 每个batch的总速率
        fairness_list: 每个batch的公平性(供对比分析, 可按需求修改计算方式)
        rates_list: 每个batch各用户的瞬时速率
    """
    batch_size, n_antennas, n_users = H_array.shape
    labels = np.zeros((batch_size, n_users), dtype=int)
    total_rates = []
    fairness_list = []
    rates_list = []

    user_rates = np.zeros(n_users)  # 每batch的用户速率

    for b in range(batch_size):
        H = H_array[b]  # shape=[n_antennas, n_users]
        selected_users = set()

        for _ in range(n_actions):
            selected_H = H[:, list(selected_users)] if selected_users else np.zeros((n_antennas, 0))
            remaining_users = [i for i in range(n_users) if i not in selected_users]

            # 若依然可加入用户(即天线数还没到上限)并且有剩余用户可选
            if selected_H.shape[1] < n_antennas and remaining_users:
                max_rate = -np.inf
                best_user = -1

                for user in remaining_users:
                    candidate_H = np.hstack((selected_H, H[:, user].reshape(-1, 1)))
                    # 若超过天线数，则跳过
                    if candidate_H.shape[1] > n_antennas:
                        continue

                    # 计算瞬时速率
                    pseudo_inverse = np.linalg.pinv(candidate_H.T @ candidate_H) @ candidate_H.T
                    a_k_H = pseudo_inverse[-1]
                    norm_a_k = np.linalg.norm(a_k_H) ** 2
                    sinr_k = p_u / norm_a_k
                    r_i_t = np.log2(1 + sinr_k)

                    # 贪婪：以瞬时速率为优先级
                    if r_i_t > max_rate:
                        max_rate = r_i_t
                        best_user = user

                # 选择速率最高的用户
                if best_user != -1:
                    selected_users.add(best_user)
                    selected_H = np.hstack((selected_H, H[:, best_user].reshape(-1, 1)))

        # 最终对已选用户计算速率
        if selected_users:
            pseudo_inverse = np.linalg.pinv(selected_H.T @ selected_H) @ selected_H.T
            for idx, user in enumerate(selected_users):
                a_k_H = pseudo_inverse[idx]
                norm_a_k = np.linalg.norm(a_k_H) ** 2
                sinr_k = p_u / norm_a_k
                user_rates[user] = np.log2(1 + sinr_k)

        # 标记已选用户
        for user in selected_users:
            labels[b, user] = 1

        # 计算总速率及公平性
        total_rate = np.sum(user_rates)
        # 简单示例的公平性: (ΣR_i)^2 / (N * Σ(R_i^2))
        fairness = (total_rate ** 2) / (n_actions * np.sum(user_rates ** 2) + 1e-8)

        total_rates.append(total_rate)
        fairness_list.append(fairness)
        rates_list.append(user_rates.copy())

        # 重置user_rates, 以免影响下一个batch
        user_rates[:] = 0.0

    return labels, total_rates, fairness_list, rates_list

def load_data_from_file(channel_path):
    data = np.load(channel_path)
    return data['H_array'], data['H_split_array'], data['user_coordinates']

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    channel_path = 'D:/python_project/GymFormimo/dataset/train/Rayleigh/uplink_30_1.npz'
    H_array, H_split_array, user_coordinates = load_data_from_file(channel_path)
    H_array = H_array[:12800]

    # 训练超参
    batch_size = 64
    n_batches = len(H_array) // batch_size
    n_actions = 8
    all_actions = 16
    p_u = 10
    t_c = 50
    n_epochs = 100
    learning_rate = 2e-2

    # 初始化模型
    model = CNN(all_actions=all_actions, device=device, actor_lr=learning_rate).to(device)
    scheduler = optim.lr_scheduler.StepLR(model.actor_optim, step_size=50, gamma=0.1)

    # ================ 训练阶段 ================
    # for epoch in range(n_epochs):
    #     epoch_loss = 0.0
    #     for batch_idx in range(n_batches):
    #         start_idx = batch_idx * batch_size
    #         end_idx = start_idx + batch_size
            
    #         H_batch = H_array[start_idx:end_idx]
    #         H_split_array_batch = H_split_array[start_idx:end_idx]
            
    #         # 生成标签
    #         label_batch, _, _, rates_list = proportional_fairness_algorithm_zf(
    #             H_batch, n_actions=n_actions, t_c=t_c, p_u=p_u
    #         )
            
    #         # 准备输入
    #         channel = torch.Tensor(H_split_array_batch).to(device)
    #         fair = torch.zeros((batch_size, all_actions)) if batch_idx == 0 else torch.Tensor(rates_list).to(device)
    #         input_data = {"channel": channel, "fair": fair}

    #         # 前向传播
    #         predictions = model(input_data)  # [batch_size, all_actions]
    #         label_batch = torch.Tensor(label_batch).to(device)  # [batch_size, all_actions]

    #         # 损失计算
    #         loss = model.compute_loss(predictions, label_batch)

    #         # 反向传播
    #         model.actor_optim.zero_grad()
    #         loss.backward()
    #         # for name, param in model.named_parameters():
    #         #     if param.grad is not None:
    #         #         print(f"Layer: {name} | Grad: {param.grad.abs().mean()}")
    #         model.actor_optim.step()
    #         # print(f"Batch {batch_idx + 1}/{n_batches}, Loss: {loss.item():.4f}")
    #         epoch_loss += loss.item()
    #     # 更新学习率
    #     scheduler.step()
    #     print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / n_batches:.4f}")

    # ================ 推理阶段 ================
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = CNN(all_actions=16, device=device, actor_lr=0.001).to(device)
    model.load_state_dict(torch.load('cnn_actor_model.pth'))
    # model.eval()  # 设置为评估模式
    # 替换成你自己的测试数据
    channel_path = 'D:/python_project/GymFormimo/dataset/train/Rayleigh/uplink_30_1.npz'
    H_array, H_split_array, user_coordinates = load_data_from_file(channel_path)

    # 取一部分做测试，这里示例取200个
    H_array_test = H_array[12800:12832]
    print("Test shape:", H_array_test.shape)

    # 准备输入数据
    input_data = {
        "channel": torch.Tensor(H_split_array[12800:12832]).to(device),
        "fair": torch.zeros((32, 1, 16)).to(device)  # 假设 fair 的输入尺寸为 (batch_size, 1, n_actions)
    }

    # 进行推理
    with torch.no_grad():
        action_logits_vec = model(input_data)
        binary_output = model.get_binary_output(action_logits_vec)

    print("Predicted action logits:", action_logits_vec)
    print("Binary output:", binary_output)

    # 保存模型
    # torch.save(model.state_dict(), "cnn_actor_model.pth")
    #
    # # 通过比例公平算法生成标签
    # labels = proportional_fairness_algorithm(H_array)
    #
    # # 定义模型
    # n_actions = 20  # 假设动作数量为 20，根据实际进行调整
    # actor_lr = 0.001
    # model = CNN(n_actions=n_actions, device=device, actor_lr=actor_lr)
    #
    # # 定义输入（这里仅作为示例，需要根据实际数据进行调整）
    # input_data = {
    #     "channel": torch.randn(32, 2, 32, 32).numpy(),  # 假设 batch_size=32, 输入尺寸为 (batch_size, 2, 32, 32)
    #     "fair": torch.randn(32, 1, n_actions).numpy()  # 假设 fair 的输入尺寸为 (batch_size, 1, n_actions)
    # }
    #
    # # 前向传播
    # action_logits_vec = model(input_data)
    # print("Predicted action logits:", action_logits_vec)
    #
    # # 计算损失
    # labels_tensor = torch.Tensor(labels).to(device)
    # loss = model.compute_loss(action_logits_vec, labels_tensor)
    # print("Loss:", loss.item())
    #
    # # 优化步骤
    # model.actor_optim.zero_grad()
    # loss.backward()
    # model.actor_optim.step()




 # # 输出部分结果
    # print("调度用户索引（部分展示）：\n", labels[:2])  # 仅展示前 2 个批次
    # print("每个批次的总速率（部分展示）：\n", total_rates[:5])  # 展示前 5 个批次的总速率
    # print("每个批次的公平性（部分展示）：\n", fairness_list[:5])  # 展示前 5 个批次的公平性
    #
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(len(total_rates)), total_rates, label="Total Rate", color="blue")
    # plt.title("Long-term Total Rate Trend")
    # plt.xlabel("Time Slot (Batch Index)")
    # plt.ylabel("Total Rate")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # # 绘制公平性变化趋势
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(len(fairness_list)), fairness_list, label="Fairness", color="green")
    # plt.title("Long-term Fairness Trend")
    # plt.xlabel("Time Slot (Batch Index)")
    # plt.ylabel("Fairness (Jain's Index)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

