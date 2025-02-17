"""
用于进行贪婪选择
"""
import torch
from Rate_ZFBF import rate_zfbf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def greedy(H, Pt):
    """
    进行贪婪选择用户组
    :param H: 所有用户信道
    :param Pt: 信噪功率比
    :return: chosen_users：选择的用户序列，chosen_SE：该序列的频谱效率
    """
    H = torch.from_numpy(H)
    number_users = H.shape[0]  # 总用户数
    number_nt = H.shape[1]  # 天线数

    chosen_users = torch.zeros(1, number_users).to(device)  # 选择的用户序列
    chosen_H = torch.Tensor()  # 选择的用户信道
    chosen_SE = torch.tensor(0).to(device)
    for i in range(number_nt):  # 最多找天线个数量的用户
        SE_pre = torch.zeros(1, number_users)  # 预选各个用户的SE
        for ii in range(number_users):  # 依次计算预选各个用户的SE
            if chosen_users[0, ii] == 0:  # 选没有选过的
                H_pre = torch.cat([chosen_H, H[ii, :]], dim=0)
                SE_pre[0, ii] = SE_pre[0, ii] + rate_zfbf(H_pre.view(-1, number_nt), Pt)
        if SE_pre.max() > chosen_SE:
            chosen_SE = SE_pre.max() * 1
            chosen_H = torch.cat([chosen_H, H[SE_pre.argmax(), :]], dim=0)
            chosen_users[0, SE_pre.argmax()] = chosen_users[0, SE_pre.argmax()] + 1
        else:
            break
    return chosen_users, chosen_SE




