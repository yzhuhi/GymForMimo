import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from gymnasium.wrappers import RecordEpisodeStatistics

# 注册环境
if __name__ == '__main__':

    # 列出所有注册的环境
    # gym.pprint_registry()

    # register(
    #     id='mimo-v0',
    #     entry_point='my_gym_env.erl_env:MimoSchedulerEnv',
    # )
    #
    # data_path = '../dataset/train/Rayleigh/uplink_40.npz'
    #
    # env = gym.make('mimo-v0', users=40, detection='ZF', pk=20, render_mode='human', channel_path=data_path)
    #
    # state, info_dict = env.reset()
    # for j in range(env.limit):
    #     action = env.action_space.sample()  # 随机动作
    #     next_state, reward, terminated, truncated, info = env.step(action)
    #     print(f"Step {j}, Reward {reward}, Terminated {terminated}, Truncated {truncated}, Info {info}")
    #     if terminated or truncated:
    #         break
    #     state = next_state
    #     env.render()
    if 'mimo_vec-v1' in gym.registry.keys():
        pass
    else:
        register(
            id='mimo_vec-v1',
            entry_point='my_gym_env.mimo:MimoSchedulerVecEnv',
        )

    # 测试查询
    data_path = '../dataset/train/Rayleigh'
    envs = gym.make_vec('mimo_vec-v1', users=40, detection="ZF", pk=20, num_envs=3, vectorization_mode='async',
                        wrappers=(gym.wrappers.RecordEpisodeStatistics,),
                       render_mode='human', channel_path=data_path, same_env=None)


    state, info = envs.reset(seed=0)
    print('selected_index', info['selected_index'])

    break_point = np.array([0] * 3)  # 统计环境退出次数
    epid = 0
    while not np.all(break_point) > 0:
        action = envs.action_space.sample()
        state, reward, terminated, truncated, _ = envs.step(action)
        if np.any(truncated) != False:
            print('terminated' + str(epid + 1), '  ', truncated)
        break_point[truncated] += 1
        # print('action'+str(epid+1)+':', action,'   ', 'reward'+str(epid+1)+':', np.round(reward,5))
        epid += 1


