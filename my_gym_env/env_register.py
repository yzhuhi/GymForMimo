import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register


if __name__ == '__main__':
    # 注册单个环境
    # register(
    #     id='mimo-v0',
    #     entry_point='my_gym_env.erl_env:MimoSchedulerEnv',
    # )
    #
    # # 测试查询
    # env = gym.spec('mimo-v0')
    # print(env)

    # 注册Vectorized环境
    # register(
    #     id='mimo-v1',
    #     entry_point='my_gym_env.vector_erl_env:MimoSchedulerVecEnv',
    # )
    # # 测试查询
    # env = gym.spec('mimo-v1')
    # print(env)
    gym.pprint_registry()

