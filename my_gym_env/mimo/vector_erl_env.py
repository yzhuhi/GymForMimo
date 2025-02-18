import json
import os
from typing import Tuple, Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

ARY = np.ndarray


class MimoSchedulerVecEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, users, detection, pk, render_mode: Optional[str] = None, channel_path: Optional[str] = None,
                 same_env: Optional[int] = None):
        super(MimoSchedulerVecEnv, self).__init__()

        # self.user_coordinates = None
        # self.H_split_array = None
        # self.H_array = None
        self.channel_list = None

        self.pk = 10 ** (pk / 10)  # db转功率 10/15/20
        self.users = users  # 用户数量
        self.limit = 300  # 最大步数
        self.limit_step = 0
        self.scheduler_num = np.zeros((1, self.users))  # 记录每个调度器被调度的次数
        self.fair_record = np.zeros(5)  # 记录每个用户的公平性

        # self.H_array, self.H_split_array, self.user_coordinates = self.load_data_from_file(channel_path)
        self.channel_path = channel_path
        self.selected_ordinate = np.array([])  # 记录当前选择的用户坐标

        # 矢量化环境，两种选择：每个环境以同一批次数据进行训练 or 每个环境都以不同批次数据进行训练
        if same_env is not None:
            self.id_choice = same_env
        else:
            self.id_choice = None

        self.time_step = 1
        self.detection = detection  # 选择的检测方式
        self.action_space = spaces.MultiBinary(40)
        self.observation_space = spaces.Dict({
            "channel": spaces.Box(low=-np.inf, high=np.inf, shape=(2, 16, self.users), dtype=np.float64),
            "fair": spaces.Box(low=0, high=np.inf, shape=(1, self.users), dtype=np.float64)
        })

        # 定义模式：人类可视化或机器人训练
        if render_mode in self.metadata["render_modes"]:

            self.render_mode = render_mode

            # 创建一个画布，设置大小和背景色
            self.fig, self.ax = plt.subplots(figsize=(10, 8))  # 增加画布大小
            self.fig.patch.set_facecolor('white')  # 设置背景色为白色

            # 设置坐标轴及其属性
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('Distance (m)', fontsize=14)  # 增加字体大小
            self.ax.set_ylabel('Distance (m)', fontsize=14)  # 增加字体大小
            self.ax.grid(True, linestyle='--', alpha=0.7)  # 设定网格样式和透明度
        elif render_mode is None or render_mode not in self.metadata["render_modes"]:
            pass

    # 显示渲染结果

    def step(self, action):
        # 与你的模拟软件（比如 CARLA）交互，获取新的观测和奖励
        # observation, reward, done = interact_with_carla(action)
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        # print(action)
        if self.time_step == self.limit - 1:
            terminated = True
        else:
            terminated = False

        se_sum, self.scheduler_num, count_pass = self.compute_se(action)

        if (self.limit_step / self.limit) > 0.2 or self.time_step == self.limit - 1:
            truncated = True
        else:
            truncated = False

        """reward"""
        reward, jain_fairness, throughput_list = self.compute_reward(se_sum, self.scheduler_num, count_pass)

        self.time_step += 1
        """next_state"""
        state = {"channel": self.H_split_array[self.time_step - 1],
                 "fair": throughput_list}

        info = {'se': se_sum, 'fair': jain_fairness}

        return state, float(reward), terminated, truncated, info

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        # 重置模拟软件并获取初始观测
        self.time_step = 1
        self.limit_step = 0
        self.scheduler_num = np.zeros((1, self.users))  # 记录每个调度器被调度的次数
        self.fair_record = np.zeros(5)  # 记录每个用户的公平性
        self.selected_ordinate = []  # 记录当前选择的用户坐标
        self.channel_list = os.listdir(self.channel_path)

        if self.id_choice is not None:
            selected_file = self.channel_list[self.id_choice]
            selected_index = self.id_choice
        else:
            selected_index = np.random.randint(0, len(self.channel_list))
            selected_file = self.channel_list[selected_index]

        selected_file = os.path.join(self.channel_path, selected_file)
        self.H_array, self.H_split_array, self.user_coordinates = self.load_data_from_file(selected_file)

        state = {"channel": self.H_split_array[self.time_step - 1],
                 "fair": self.scheduler_num}

        info_dict = {'selected_index': selected_index}

        return state, info_dict

    def render(self):
        if self.render_mode == "human":
            # 如果是第一次绘制图像，创建散点图等对象
            if not hasattr(self, 'user_scatter'):
                self.ax.clear()

                # 通信范围
                R = 100  # 基站覆盖半径
                d0 = 1
                a, b = 0, 0  # 基站坐标
                theta = np.linspace(0, 2 * np.pi, 200)  # 绘图用的角度
                X = a + R * np.cos(theta)
                Y = b + R * np.sin(theta)

                # 绘制通信范围
                self.ax.fill(X, Y, color='blue', alpha=0.1, label='Communication Range')
                self.ax.plot(0, 0, marker='^', color='black', markersize=10, label='Base Station')
                self.ax.plot(d0 * np.cos(theta), d0 * np.sin(theta), '--g', linewidth=2, label='Reference Range')

                # 初始化用户位置散点图
                user_x = [coord[0] for coord in self.user_coordinates[self.time_step - 2]]
                user_y = [coord[1] for coord in self.user_coordinates[self.time_step - 2]]
                self.user_scatter = self.ax.scatter(user_x, user_y, color='red', label='User Positions', s=50,
                                                    alpha=0.6)

                # 初始化选定用户位置散点图
                self.selected_user_scatter = self.ax.scatter([], [], color='magenta', label='Selected User Positions',
                                                             s=50, alpha=0.8)

                # 添加用户注释
                for i in range(self.users):
                    self.ax.annotate(f'UE{i + 1}', (user_x[i], user_y[i]), fontsize=8, ha='right', color='black')

                # 设置图形属性
                self.ax.set_aspect('equal')
                self.ax.set_xlabel('Distance (m)', fontsize=12)
                self.ax.set_ylabel('Distance (m)', fontsize=12)
                self.ax.set_xlim(-(R + 20), R + 20)
                self.ax.set_ylim(-(R + 20), R + 20)
                self.ax.grid(True, linestyle='--', alpha=0.7)
                self.ax.legend(loc='upper right')

            # 更新用户位置
            user_x = [coord[0] for coord in self.user_coordinates[self.time_step - 2]]
            user_y = [coord[1] for coord in self.user_coordinates[self.time_step - 2]]
            self.user_scatter.set_offsets(np.c_[user_x, user_y])

            # 更新被选中的用户位置
            if len(self.selected_ordinate) > 0:
                user_x1 = [coord1[0] for coord1 in self.selected_ordinate[self.time_step - 2]]
                user_y1 = [coord1[1] for coord1 in self.selected_ordinate[self.time_step - 2]]
                self.selected_user_scatter.set_offsets(np.c_[user_x1, user_y1])

            # 刷新显示
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.1)

    def get_normal_value(self, se_sum):
        # 将 se_sum 转换为字符串
        se_sum_str = str(int(se_sum))  # 转换为整数再转为字符串以去掉小数点
        # 获取位数
        num_digits = len(se_sum_str)
        # 计算最高位的权值
        highest_value = 10 ** (num_digits - 1)
        return se_sum / highest_value

    def count_trailing_zeros(self, value):
        # 将值转为字符串
        str_value = str(value)

        # 如果没有小数部分，直接返回0
        if '.' not in str_value:
            return 0

        # 分割小数部分
        decimal_part = str_value.split('.')[1]

        # 计算小数点后0的个数
        trailing_zeros = len(decimal_part) - len(decimal_part.rstrip('0'))

        return trailing_zeros

    def compute_reward(self, se_sum, scheduler_num, count_pass):
        # 计算fairness reward
        throughput_list = scheduler_num / self.time_step
        if np.sum(throughput_list ** 2) == 0:
            jain_fairness = 0  # 或者根据需求设定其他默认值
        else:
            jain_fairness = np.sum(throughput_list) ** 2 / (np.sum(throughput_list ** 2) * self.users)
        # 进行记录的平移操作
        # if len(self.fair_record) < 5:
        #     raise ValueError("fair_record长度不足，无法进行更新。")

        self.fair_record[:-1] = self.fair_record[1:]  # 移动值到前面
        self.fair_record[-1] = jain_fairness  # 更新最后一个值

        if np.unique(self.fair_record).size == 1:
            fair_reward = -2.0
        else:
            fair_reward = jain_fairness


        # 计算SE reward
        step_reward = 0.5
        normalized_step_reward = self.get_normal_value(se_sum)
        se_fair_reward = jain_fairness * normalized_step_reward

        # 回合结束奖励 or 截断惩罚
        if self.time_step == self.limit - 1:
            end_reward = 10.0
        elif (self.limit_step / self.limit) > 0.2:
            end_reward = -5.0
        else:
            end_reward = 0.0

        # 计算惩罚
        if count_pass != 0:
            # percent = 16 / count_pass
            step_reward, fair_reward, se_fair_reward = 0, -jain_fairness, -se_fair_reward

        reward = sum([step_reward, fair_reward, se_fair_reward, end_reward])

        return reward, jain_fairness, throughput_list

    def compute_se(self, action, se_sum=0):
        # 计算SE
        selected_users = []
        p_sum = 0
        count = 0
        count_pass = 0
        for i, act in enumerate(action):
            if act:
                if self.detection == "ZF":
                    W = (self.H_array[self.time_step - 1, :, i:i + 1] @ np.linalg.inv(
                        self.H_array[self.time_step - 1, :, i:i + 1].T.conjugate() @ self.H_array[self.time_step - 1, :,
                                                                                     i:i + 1]))
                    # analog MRC
                elif self.detection == "dMRC":
                    W = self.H_array[self.time_step - 1, :, i:i + 1]
                elif self.detection == "aMRC":
                    angles = np.angle(self.H_array[self.time_step - 1, :, i:i + 1])
                    W = np.exp(1j * angles)

                users_list = list(range(self.users))
                ue_list = [ue for ue in users_list if ue != i]
                for user in ue_list:
                    p_sum += (np.abs(np.dot(W.T.conjugate(), self.H_array[self.time_step - 1, :, user:user + 1]))) ** 2

                if self.detection == "ZF":
                    p_sum = 0

                sinr = self.pk * (
                        (np.abs((np.dot(W.T.conjugate(), self.H_array[self.time_step - 1, :, i:i + 1])))) ** 2) / (
                               ((np.linalg.norm(W.T.conjugate())) ** 2) + (p_sum * self.pk))

                se_sum += np.log2(1 + sinr.squeeze())
                self.scheduler_num[0][i] += 1
                count += 1
                selected_users.append(i)
        if count > 16:
            self.limit_step += 1
            count_pass = count
        self.selected_ordinate.append(self.user_coordinates[self.time_step - 1][selected_users])
        return se_sum, self.scheduler_num, count_pass

    def load_data_from_file(self, file_path):
        data = np.load(file_path)

        # 从npz文件中获取数据
        H_array = data['H_array']
        H_split_array = data['H_split_array']
        user_coordinates = data['user_coordinates']

        return H_array, H_split_array, user_coordinates


if __name__ == '__main__':
    data_path = '/dataset/train/Rayleigh'
    env = MimoSchedulerVecEnv(users=40, detection="ZF", pk=20, render_mode='human', channel_path=data_path)
    np.random.seed(2)  # 固定随机种子

    for i in range(4):
        state, info_dict = env.reset()
        print(info_dict["selected_index"])
        for j in range(env.limit):
            action = env.action_space.sample()  # 随机动作
            next_state, reward, terminated, truncated, info = env.step(action)
            # print(f"Step {j}, Reward {reward}, Terminated {terminated}, Truncated {truncated}, Info {info}")
            if terminated or truncated:
                break
            state = next_state
            # env.render()

    # from gym.envs.registration import register
    #
    # register(
    #     id='AutoDriving-v0',
    #     entry_point='your_module:AutoDrivingEnv',
    # )
    #
    # import gym
    # env = gym.make('AutoDriving-v0')
