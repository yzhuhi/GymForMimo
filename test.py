import numpy as np
from gymnasium.envs.registration import register
import gymnasium as gym
import pandas as pd
from a2c import A2C
import torch

if __name__ == '__main__':

    if "mimo_vec-v1" not in gym.registry.keys():
        register(
            id='mimo_vec-v1',
            entry_point='my_gym_env.mimo:MimoSchedulerVecEnv',
        )

    # 测试查询
    data_path = 'dataset/test/Rayleigh'
    gym.pprint_registry()
    # environment hyperparams
    num_envs = 1
    n_updates = 2000
    n_steps_per_update = 128
    actor_lr = 1e-3
    critic_lr = 0.006
    obs_shape = 656
    action_shape = 40
    n_showcase_episodes = 1

    envs = gym.make_vec('mimo_vec-v1', users=40, detection="ZF", pk=20, num_envs=1,
                        render_mode='human', channel_path=data_path, same_env=None)

    # set the device
    use_cuda = True
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # load the agent

    agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, num_envs, traing=False)

    # 加载参数
    agent.load_state_dict(torch.load('weights/a2c_best_model.pth', weights_only=True))

    agent.eval()  # 设置为评估模式
    # 加载之后
    # print("----------------------------")
    # for name, param in agent.named_parameters():
    #     print(name, param)

    for episode in range(n_showcase_episodes):
        print(f"starting episode {episode}...")

        # get an initial state
        state, info = envs.reset()

        # 初始化累积值和计数器
        se_sum = 0
        step_count = 0

        # play one episode
        done = False
        while not done:
            # select an action A_{t} using S_{t} as input for the agent
            with torch.no_grad():
                actions, action_log_probs, state_value_preds, entropy = agent.select_action(state)

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            print(actions)
            next_state, reward, terminated, truncated, info = envs.step(actions.cpu().numpy())

            se_sum += info['se']
            step_count += 1

            # update if the environment is done
            if terminated or truncated:
                break
            state = next_state
            envs.render()
        # 计算平均值
        print(step_count)
        average_se = se_sum / step_count if step_count > 0 else 0
        # average_fairness = fairness_sum / step_count if step_count > 0 else 0

        # 输出平均值
        print(f"平均 SE: {average_se}, 平均公平性: {info['fair']}")
