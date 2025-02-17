"""
用于测试多个TTI下的用户选择，采用的cnn，考虑jfi
"""

import numpy as np  # 引入NumPy包，代替matlab的矩阵运算功能
import torch
import time
from Functions import H2Network, JFI, act_with_jfi, choose, epsilon_greedy
from tqdm import tqdm
from greedy import greedy
import matplotlib.pyplot as plt
from Rayleigh_Channel import RayleighChannel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  参数预设
Nt = 16
N_users = 30
number_actions = N_users + 1

snr = np.array([-15, -10, -5, 0, 5, 10, 15])

epochs = 3  # 测试轮次
TTIs = 60  # 选择次数

"""
常用
print(type(state0))
print(state0.shape)
print(state0.dtype)
"""

SE = torch.zeros(7, epochs, TTIs).to(device)  # 每次epoch的每个TTI的SE
jfi = torch.zeros(7, epochs).to(device)  # 每次epoch的总jfi

# 加载信道 3*60*[(30or40)*16]
H = np.load('./data_for_test/30/Rayleigh_30_3_60.npy')  # array类型，complex128(double*2)，3*60*(N_users*Nt)

for i_snr in tqdm(range(7)):
    #SNR = 100
    SNR = i_snr * 5 - 15
    Pt = np.power(10, SNR / 10)
    for epo in range(epochs):
        chosen_users_all = torch.zeros(1, N_users).to(device)
        for tti in range(TTIs):
            H_30 = H[(epo*60+tti)*30:(epo*60+tti+1)*30, :]  # array类型，complex128(double*2)，N_users*Nt
            chosen_users, chosen_SE = greedy(H_30, Pt)
            chosen_users_all = chosen_users_all + chosen_users
            SE[i_snr, epo, tti] = SE[i_snr, epo, tti] + chosen_SE
        jfi[i_snr, epo] = jfi[i_snr, epo] + JFI(chosen_users_all)

print(SE[6, :, :])
SE_snr_epo_mean = torch.mean(SE, dim=2)  # 计算不同snr下三个epoch的每个TTI的SE均值
print(SE_snr_epo_mean[6, :])
SE_snr_mean = torch.mean(SE_snr_epo_mean, dim=1, keepdim=True)  # 计算不同snr下三个epoch的SE均值

jfi_snr_mean = torch.mean(jfi, dim=1, keepdim=True)  # 计算不同snr下三个epoch的jfi均值
print(jfi_snr_mean)

# 保存训练的每次epoch的jfi
jfi = jfi.to(torch.device('cpu')).detach().numpy()
np.save('./result/test/30/Greedy/jfi.npy', jfi)

# 保存不同snr下三个epoch的每个TTI的SE
SE = SE.to(torch.device('cpu')).detach().numpy()
np.save('./result/test/30/Greedy/SE.npy', SE)

# 保存不同snr下三个epoch的SE均值
SE_snr_mean = SE_snr_mean.to(torch.device('cpu')).detach().numpy()
np.save('./result/test/30/Greedy/SE_snr_mean.npy', SE_snr_mean)

# 画SE_snr_mean图
plt.plot(snr, SE_snr_mean)
plt.xlabel('SNR')
plt.ylabel('SE')
# 保存训练图像
plt.savefig('./result/test/30/Greedy/SE_snr_mean.png', dpi=300, format='png')
plt.show()

# 画jfi图
jfi_snr_mean = jfi_snr_mean.to(torch.device('cpu')).detach().numpy()
plt.plot(snr, jfi_snr_mean)
plt.xlabel('SNR')
plt.ylabel('jfi')
# 保存训练图像
plt.savefig('./result/test/30/Greedy/jfi_snr_mean.png', dpi=300, format='png')
plt.show()
