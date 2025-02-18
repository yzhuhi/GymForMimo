import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
data = pd.read_csv('./episode_returns.csv')

# 提取数据
episodes = data['Episodes'][:500]
returns = data['Returns'][:500]

# 绘制图形
plt.figure(figsize=(10, 5))
plt.plot(episodes, returns, label='Returns')
plt.xlabel('Number of episodes')
plt.ylabel('Returns')
plt.title('Returns Over Episodes')
plt.legend()
plt.grid()
plt.show()
