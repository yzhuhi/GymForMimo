import json
from typing import Tuple, Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ARY = np.ndarray


class PendulumEnv(gym.Wrapper):  # a demo of custom env
    def __init__(self):
        gym_env_name = 'Pendulum-v1'
        super().__init__(env=gym.make(gym_env_name))

        '''the necessary env information when you design a custom env'''
        self.env_name = gym_env_name  # the name of this env.
        self.state_dim = self.observation_space.shape[0]  # feature number of state
        self.action_dim = self.action_space.shape[0]  # feature number of action
        self.if_discrete = False  # discrete action or continuous action

    def reset(self, **kwargs) -> Tuple[ARY, dict]:  # reset the agent in env
        state, info_dict = self.env.reset()
        return state, info_dict

    def step(self, action: ARY) -> Tuple[ARY, float, bool, bool, dict]:  # agent interacts in env
        # OpenAI Pendulum env set its action space as (-2, +2). It is bad.
        # We suggest that adjust action space to (-1, +1) when designing a custom env.
        state, reward, terminated, truncated, info_dict = self.env.step(action * 2)
        state = state.reshape(self.state_dim)
        return state, float(reward) * 0.5, terminated, truncated, info_dict


from typing import Optional
# from mimo_rayleigh_lowv1 import load_data_from_file

class MimoSchedulerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, users, detection, pk, render_mode: Optional[str] = None, channel_path: Optional[str] = None):
        super(MimoSchedulerEnv, self).__init__()

        self.pk = 10**(pk / 10) # db转功率 10/15/20
        self.users = users  # 用户数量
        self.limit = 600
        self.limit_step = 0
        self.scheduler_num = np.zeros(self.users)  # 记录每个调度器被调度的次数
        self.fair_record = np.zeros(2)  # 记录每个用户的公平性
        self.H_array, self.H_split_array, self.user_coordinates = self.load_data_from_file(channel_path)
        self.selected_ordinate = np.array([])  # 记录当前选择的用户坐标

        self.time_step = 1
        self.detection = detection  # 选择的检测方式
        self.action_space = spaces.MultiBinary(40)
        self.observation_space = spaces.Dict({
            "channel": spaces.Box(low=-np.inf, high=np.inf, shape=(2, 16, self.users), dtype=np.float64),
            "fair": spaces.Box(low=0, high=np.inf, shape=(1, self.users), dtype=np.float64)
        })

        # 定义模式：人类可视化或机器人训练
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {render_mode}")
        else:
            self.render_mode = render_mode

            # 创建一个画布，设置大小和背景色
            self.fig, self.ax = plt.subplots(figsize=(10, 8))  # 增加画布大小
            self.fig.patch.set_facecolor('white')  # 设置背景色为白色

            # 设置坐标轴及其属性
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('Distance (m)', fontsize=14)  # 增加字体大小
            self.ax.set_ylabel('Distance (m)', fontsize=14)  # 增加字体大小
            self.ax.grid(True, linestyle='--', alpha=0.7)  # 设定网格样式和透明度

    # 显示渲染结果

    def step(self, action):
        # 与你的模拟软件（比如 CARLA）交互，获取新的观测和奖励
        # observation, reward, done = interact_with_carla(action)
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        if self.time_step > self.limit:
            terminated = True
        else:
            terminated = False

        se_sum, self.scheduler_num, count_pass = self.compute_se(action)


        if self.limit_step > (self.limit / 2):
            truncated = True
        else:
            truncated = False

        """reward"""
        reward, jain_fairness, throughput_list = self.compute_reward(se_sum, self.scheduler_num, count_pass, terminated, truncated)

        self.time_step += 1
        """next_state"""
        state = {"channel": self.H_split_array[self.time_step],
                 "fair": throughput_list}

        info = {'se': se_sum, 'fair': jain_fairness}

        return state, float(reward), terminated, truncated, info

    def reset(self, **kwargs) -> tuple[Any, dict[Any, Any]]:
        # 重置模拟软件并获取初始观测
        self.time_step = 1
        self.limit_step = 0
        self.scheduler_num = np.zeros(self.users)  # 记录每个调度器被调度的次数
        self.fair_record = np.zeros(2)  # 记录每个用户的公平性
        self.selected_ordinate = []   # 记录当前选择的用户坐标
        state = {"channel": self.H_split_array[self.time_step-1],
                 "fair": self.scheduler_num}
        info_dict = {}

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
            plt.pause(1)

    def get_normal_value(self, se_sum):
        # 将 se_sum 转换为字符串
        se_sum_str = str(int(se_sum))  # 转换为整数再转为字符串以去掉小数点
        # 获取位数
        num_digits = len(se_sum_str)
        # 计算最高位的权值
        highest_value = 10 ** (num_digits - 1)
        return se_sum / highest_value

    def compute_reward(self, se_sum, scheduler_num, count_pass, terminated=None, tuncated=None):
        # 计算fairness reward
        throughput_list = scheduler_num / self.time_step
        jain_fairness = np.sum(throughput_list ** 2) / (np.sum(throughput_list) ** 2)
        self.fair_record[0] = self.fair_record[1]  # 将之前的值移到前面
        self.fair_record[1] = jain_fairness
        fair_reward = self.fair_record[1] - self.fair_record[0]
        # 计算SE reward
        step_reward = 1.0
        normalized_step_reward = self.get_normal_value(se_sum)
        se_fair_reward = jain_fairness * normalized_step_reward
        # 计算惩罚
        if count_pass:
            step_reward, fair_reward, se_fair_reward = 0, 0, 0
        # 回合结束奖励 or 截断惩罚
        if terminated:
            step_reward += 10.0
        elif tuncated:
            step_reward -= 10.0

        reward = sum([step_reward, fair_reward, se_fair_reward])

        return reward, jain_fairness, throughput_list

    def compute_se(self, action, se_sum=0):
        # 计算SE
        selected_users = []
        p_sum = 0
        count = 0
        count_pass = False
        for i, act in enumerate(action):
            if act:
                if self.detection == "ZF":
                    W = (self.H_array[self.time_step, :, i:i + 1] @ np.linalg.inv(
                        self.H_array[self.time_step, :, i:i + 1].T.conjugate() @ self.H_array[self.time_step, :, i:i + 1]))
                    # analog MRC
                elif self.detection == "dMRC":
                    W = self.H_array[self.time_step, :, i:i + 1]
                elif self.detection == "aMRC":
                    angles = np.angle(self.H_array[self.time_step, :, i:i + 1])
                    W = np.exp(1j * angles)

                users_list = list(range(self.users))
                ue_list = [ue for ue in users_list if ue != i]
                for user in ue_list:
                    p_sum += (np.abs(np.dot(W.T.conjugate(), self.H_array[self.time_step, :, user:user + 1]))) ** 2

                if self.detection == "ZF":
                    p_sum = 0

                sinr = self.pk * ((np.abs((np.dot(W.T.conjugate(), self.H_array[self.time_step, :, i:i + 1])))) ** 2) / (
                            ((np.linalg.norm(W.T.conjugate())) ** 2) + (p_sum * self.pk))
    
                se_sum += np.log2(1 + sinr.squeeze())
                self.scheduler_num[i] += 1
                count += 1
                selected_users.append(i)
        if count > 16:
            self.limit_step += 1
            count_pass = True
        self.selected_ordinate.append(self.user_coordinates[self.time_step-1][selected_users])
        return se_sum, self.scheduler_num, count_pass

    def load_data_from_file(self,file_path):
        data = np.load(file_path)

        # 从npz文件中获取数据
        H_array = data['H_array']
        H_split_array = data['H_split_array']
        user_coordinates = data['user_coordinates']

        return H_array, H_split_array, user_coordinates


if __name__ == '__main__':
    data_path = 'D:/python_project/ElegantRL-master/hellomimo/dataset/train/Rayleigh/uplink_40.npz'
    env = MimoSchedulerEnv(users=40, detection="ZF", pk=20, render_mode='human', channel_path=data_path)

    state, info_dict = env.reset()
    for j in range(env.limit):
        action = env.action_space.sample()   # 随机动作
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {j}, Reward {reward}, Terminated {terminated}, Truncated {truncated}, Info {info}")
        if terminated or truncated:
            break
        state = next_state
        env.render()



    # from gym.envs.registration import register
    #
    # register(
    #     id='AutoDriving-v0',
    #     entry_point='your_module:AutoDrivingEnv',
    # )
    #
    # import gym
    # env = gym.make('AutoDriving-v0')
