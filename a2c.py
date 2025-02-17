from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from gymnasium.envs.registration import register
import gymnasium as gym
import pandas as pd


class A2C(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
            self,
            n_features: int,
            n_actions: int,
            device: torch.device,
            critic_lr: float,
            actor_lr: float,
            n_envs: int,
            traing: bool = True,
            epsilon: float = 0.4,
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.traing = traing
        self.epsilon = epsilon

        # 定义 critic 的层
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3, 3), stride=1),  # 第一次卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 第一次池化

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1),  # 第二次卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 第二次池化

            nn.Flatten()  # 展平输入
        ).to(self.device)

        self.critic_mlp = nn.Sequential(
            nn.Linear(32 * 2 * (n_actions - 8) // 4, 32),
            nn.ReLU()
        ).to(self.device)

        # MLP 用于 fair 输入
        self.critic_fair_mlp = nn.Sequential(
            nn.Flatten(),  # 展平输入
            nn.Linear(1 * n_actions, 32),  # fair 输入的 MLP 层
            nn.ReLU()
        ).to(self.device)

        self.critic_last_layer = nn.Sequential(nn.Linear(64, 32),
                                               nn.ReLU(),
                                               nn.Linear(32, 1)).to(self.device)  # 输出 V(s)

        # 定义 actor 的层
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3, 3), stride=1),  # 第一次卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 第一次池化

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1),  # 第二次卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 第二次池化

            nn.Flatten()  # 展平输入
        ).to(self.device)

        self.actor_mlp = nn.Sequential(
            nn.Linear(32 * 2 * (n_actions - 8) // 4, 64),  # 更新为适当的输入尺寸
            nn.ReLU()
        ).to(self.device)

        self.actor_fair_mlp = nn.Sequential(
            nn.Flatten(),  # 展平输入
            nn.Linear(1 * n_actions, 32),  # fair 输入的 MLP 层
            nn.ReLU()
        ).to(self.device)

        self.actor_last_layer = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Sigmoid()
        ).to(self.device)

        # define optimizers for actor and critic
        # 定义优化器，传入 Critic 和 Actor 的所有参数
        self.critic_optim = optim.Adam(
            list(self.critic_cnn.parameters()) +
            list(self.critic_mlp.parameters()) +
            list(self.critic_fair_mlp.parameters()) +
            list(self.critic_last_layer.parameters()),
            lr=critic_lr
        )

        self.actor_optim = optim.Adam(
            list(self.actor_cnn.parameters()) +
            list(self.actor_mlp.parameters()) +
            list(self.actor_fair_mlp.parameters()) +
            list(self.actor_last_layer.parameters()),
            lr=actor_lr
        )

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        channel = x["channel"]
        fair = x["fair"]
        channel = torch.Tensor(channel).to(self.device)
        fair = torch.Tensor(fair).to(self.device)
        # x = torch.Tensor(x).to(self.device)
        # 处理 channel
        c1 = self.critic_cnn(channel)
        c2 = self.critic_mlp(c1)
        # 处理 fair
        f1 = self.critic_fair_mlp(fair)
        # 合并 channel 和 fair
        combined_critic_features = torch.cat((c2, f1), dim=1)
        # 计算 state_values
        state_values = self.critic_last_layer(combined_critic_features)

        a1 = self.actor_cnn(channel)
        a2 = self.actor_mlp(a1)
        # 处理 fair
        f2 = self.actor_fair_mlp(fair)
        # 合并 channel 和 fair
        combined_actor_features = torch.cat((a2, f2), dim=1)
        action_logits_vec = self.actor_last_layer(combined_actor_features)

        # state_values = self.critic(x)  # shape: [n_envs,]
        # action_logits_vec = self.actor(x)  # shape: [n_envs, n_actions]
        return state_values, action_logits_vec

    def select_action(
            self, x: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
        # 在每次调用函数时更新 epsilon
        if self.traing:
            self.epsilon = self.epsilon * 0.99

        state_values, action_logits = self.forward(x)

        if self.traing and np.random.rand() < self.epsilon:
            # 以 ε 的概率随机选择动作
            action_pd = torch.distributions.Bernoulli(probs=action_logits)  # 用 Bernoulli 分布进行采样
            actions = action_pd.sample()
            action_probs = actions * action_logits + (1 - actions)  # 一个很小的正数
            entropy = action_pd.entropy()
        else:
            actions = (action_logits > 0.5).float()
            action_probs = actions * action_logits + (1 - actions)  # 一个很小的正数
            entropy = -torch.sum(action_probs * torch.log(action_probs), dim=1).mean()

        action_probs_product = torch.prod(action_probs, dim=1)
        action_log_probs = torch.log(action_probs_product)

        return actions, action_log_probs, state_values, entropy

    def get_losses(
            self,
            rewards: torch.Tensor,
            action_log_probs: torch.Tensor,
            value_preds: torch.Tensor,
            entropy: torch.Tensor,
            masks: torch.Tensor,
            gamma: float,
            lam: float,
            ent_coef: float,
            device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                    rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
                -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update_parameters(
            self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


if __name__ == "__main__":

    if "mimo_vec-v1" not in gym.registry.keys():
        register(
            id='mimo_vec-v1',
            entry_point='my_gym_env.mimo:MimoSchedulerVecEnv',
        )
    # 测试查询
    data_path = 'dataset/train/Rayleigh'
    gym.pprint_registry()
    # environment hyperparams
    num_envs = 4
    n_updates = 2000
    n_steps_per_update = 128
    randomize_domain = False

    # agent hyperparams
    gamma = 0.999
    lam = 0.95  # hyperparameter for GAE
    ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
    actor_lr = 0.001
    critic_lr = 0.005

    # envs = gym.vector.make('mimo_vec-v1', users=40, detection="ZF", pk=20, num_envs=n_envs,
    #                        render_mode="None", channel_path=data_path, same_env=None)

    envs = gym.make_vec('mimo_vec-v1', users=40, detection="ZF", pk=20, num_envs=num_envs,
                        vectorization_mode='async',
                        render_mode=None, channel_path=data_path, same_env=None)

    envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(envs, buffer_length=num_envs * n_updates)

    obs_shape = 656
    action_shape = 40

    # set the device
    use_cuda = True
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # init the agent
    agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, num_envs)

    # create a wrapper environment to save episode returns and episode lengths
    # envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=num_envs * n_updates)

    critic_losses = []
    actor_losses = []
    entropies = []
    max_return = -float('inf')  # 初始化为负无穷大

    # use tqdm to get a progress bar for training
    for sample_phase in tqdm(range(n_updates)):
        # we don't have to reset the envs, they just continue playing
        # until the episode is over and then reset automatically

        # reset lists that collect experiences of an episode (sample phase)
        ep_value_preds = torch.zeros(n_steps_per_update, num_envs, device=device)
        ep_rewards = torch.zeros(n_steps_per_update, num_envs, device=device)
        ep_action_log_probs = torch.zeros(n_steps_per_update, num_envs, device=device)
        masks = torch.zeros(n_steps_per_update, num_envs, device=device)

        # at the start of training reset all envs to get an initial state
        if sample_phase == 0:
            states, info = envs_wrapper.reset(seed=42)

        # play n steps in our parallel environments to collect data
        for step in range(n_steps_per_update):
            # select an action A_{t} using S_{t} as input for the agent
            actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                states
            )

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            states, rewards, terminated, truncated, infos = envs_wrapper.step(
                actions.cpu().numpy()
            )

            ep_value_preds[step] = torch.squeeze(state_value_preds)
            ep_rewards[step] = torch.tensor(rewards, device=device)
            ep_action_log_probs[step] = action_log_probs

            # add a mask (for the return calculation later);
            # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
            masks[step] = torch.tensor([not term for term in terminated])

        # calculate the losses for actor and critic
        critic_loss, actor_loss = agent.get_losses(
            ep_rewards,
            ep_action_log_probs,
            ep_value_preds,
            entropy,
            masks,
            gamma,
            lam,
            ent_coef,
            device,
        )

        # update the actor and critic networks
        agent.update_parameters(critic_loss, actor_loss)

        # log the losses and entropy
        critic_losses.append(critic_loss.detach().cpu().numpy())
        actor_losses.append(actor_loss.detach().cpu().numpy())
        entropies.append(entropy.detach().mean().cpu().numpy())

        # 计算当前的 episode_returns_moving_average
        current_episode_return_moving_average = (
                np.convolve(np.array(envs_wrapper.return_queue).flatten(),
                            np.ones(10),
                            mode="valid") / 10
        )

        # 检查当前的平均回报是否大于历史最大回报
        if current_episode_return_moving_average[-1] > max_return:
            max_return = current_episode_return_moving_average[-1]
            # 保存模型
            try:
                torch.save(agent.state_dict(), 'weights/a2c_best_model_2.pth')
                print(f"模型已保存，当前最高平均回报为: {max_return}")
            except Exception as e:
                print(f"保存模型时出错: {e}")


    """ plot the results """
    # %matplotlib inline

    rolling_length = 20
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
    fig.suptitle(
        f"Training plots for {agent.__class__.__name__} in the mimo_vec-v1 environment \n \
                 (n_envs={num_envs}, n_steps_per_update={n_steps_per_update}, randomize_domain={randomize_domain})"
    )

    # episode return
    axs[0][0].set_title("Episode Returns")
    episode_returns_moving_average = (
            np.convolve(
                np.array(envs_wrapper.return_queue).flatten(),
                np.ones(rolling_length),
                mode="valid",
            )
            / rolling_length
    )
    # 记录长度数据
    length_queue = np.array(envs_wrapper.length_queue).flatten()
    length_moving_average = (
            np.convolve(
                length_queue,
                np.ones(rolling_length),
                mode="valid",
            )
            / rolling_length
    )
    # 记录数据到 CSV 文件
    episodes = np.arange(len(episode_returns_moving_average)) / num_envs
    data_to_save = pd.DataFrame({
        "Episodes": episodes,
        "Returns": episode_returns_moving_average,
        "Lengths": length_moving_average[:len(episode_returns_moving_average)]  # 对应长度
    })
    # 指定保存路径
    save_path = 'record_returns/episode_returns_2.csv'

    # 创建文件夹（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存数据
    data_to_save.to_csv(save_path, index=False)
    # 绘制曲线
    axs[0][0].plot(
        np.arange(len(episode_returns_moving_average)) / num_envs,
        episode_returns_moving_average,
    )
    axs[0][0].set_xlabel("Number of episodes")

    # entropy
    axs[1][0].set_title("Entropy")
    entropy_moving_average = (
            np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
            / rolling_length
    )
    axs[1][0].plot(entropy_moving_average)
    axs[1][0].set_xlabel("Number of updates")

    # critic loss
    axs[0][1].set_title("Critic Loss")
    critic_losses_moving_average = (
            np.convolve(
                np.array(critic_losses).flatten(), np.ones(rolling_length), mode="valid"
            )
            / rolling_length
    )
    axs[0][1].plot(critic_losses_moving_average)
    axs[0][1].set_xlabel("Number of updates")

    # actor loss
    axs[1][1].set_title("Actor Loss")
    actor_losses_moving_average = (
            np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid")
            / rolling_length
    )
    axs[1][1].plot(actor_losses_moving_average)
    axs[1][1].set_xlabel("Number of updates")

    plt.tight_layout()
    # 保存图形到文件
    save_fig_path = 'record_returns/training_plots_2.png'  # 图形保存路径
    plt.savefig(save_fig_path)

    plt.show()

    """ save model """
    save_weights = True
    load_weights = False

    """ save network weights """
    if save_weights:
        try:
            # Save model
            torch.save(agent.state_dict(), 'weights/a2c_model_2.pth')
        except Exception as e:
            print(f"保存模型时出错: {e}")

    """ load network weights """
    if load_weights:
        try:
            agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, num_envs)
            agent.load_state_dict(torch.load('weights/a2c_model_2.pth'))
        except FileNotFoundError:
            print("模型文件未找到，请检查路径。")
        except Exception as e:
            print(f"加载模型时出错: {e}")
