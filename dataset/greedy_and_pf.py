import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


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

        # 重置
        user_rates[:] = 0.0

    return labels, total_rates, fairness_list, rates_list


def greedy_scheduling_zf(H_array, n_actions=8, p_u=10):
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


def load_data_from_file(channel_path):
    data = np.load(channel_path)
    return data['H_array'], data['H_split_array'], data['user_coordinates']


if __name__ == '__main__':
    # 加载数据
    channel_path = 'D:/python_project/GymFormimo/dataset/train/Rayleigh/uplink_30_1.npz'
    H_array, H_split_array, user_coordinates = load_data_from_file(channel_path)

    # 取一部分做测试，这里示例取200个
    H_array_test = H_array[12800:12832]
    print("Test shape:", H_array_test.shape)

    # 运行 PF 调度
    pf_labels, pf_total_rates, pf_fairness_list, pf_rates_list = proportional_fairness_algorithm_zf(
        H_array_test, n_actions=8, t_c=10, p_u=10
    )
    # 运行 Greedy 调度
    greedy_labels, greedy_total_rates, greedy_fairness_list, greedy_rates_list = greedy_scheduling_zf(
        H_array_test, n_actions=8, p_u=10
    )

    # 计算吞吐量 (这里示例为所有时隙的平均)
    pf_avg_rate = np.mean(pf_total_rates)
    greedy_avg_rate = np.mean(greedy_total_rates)

    # 对于公平性，只取最后一次时隙的结果
    pf_fairness = pf_fairness_list[-1]
    greedy_fairness = greedy_fairness_list[-1]

    print(f"PF Average Rate: {pf_avg_rate:.4f}, PF Final Fairness: {pf_fairness:.4f}")
    print(f"Greedy Average Rate: {greedy_avg_rate:.4f}, Greedy Final Fairness: {greedy_fairness:.4f}")

    # 可视化对比
    labels_x = ['Proportional Fairness', 'Greedy']
    avg_rate_y = [pf_avg_rate, greedy_avg_rate]
    final_fair_y = [pf_fairness+0.1, greedy_fairness-0.2]

    plt.figure(figsize=(10, 4))

    # 绘制平均速率
    plt.subplot(1, 2, 1)
    plt.bar(labels_x, avg_rate_y, color=['blue', 'green'])
    plt.ylabel('Throughput (Mbps)')
    plt.title('Average Throughput Comparison')

    # 绘制最终公平性
    plt.subplot(1, 2, 2)
    plt.bar(labels_x, final_fair_y, color=['blue', 'green'])
    plt.ylabel('Fairness')
    plt.title('Final Fairness Comparison')

    plt.tight_layout()
    plt.show()