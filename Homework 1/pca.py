import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. 读取数据并分离输入与输出
df = pd.read_csv('Concrete_Data_Yeh.csv')
X = df.iloc[:, :8].values  # 原始 8 个输入特征
y = df.iloc[:, 8].values   # 输出：抗压强度

# 2. 数据标准化 (PCA 的关键前置步骤)
# 确保每个特征的均值为 0，方差为 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 拟合 PCA 模型 (初始保留所有 8 个成分以观察方差分布)
pca = PCA(n_components=8)
pca.fit(X_scaled)

# 4. 获取方差解释率
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print("=== 各主成分的方差解释率 ===")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"主成分 {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

print("\n=== 累积方差解释率 ===")
for i, cum_ratio in enumerate(cumulative_variance_ratio):
    print(f"前 {i+1} 个主成分累积解释: {cum_ratio:.4f} ({cum_ratio*100:.2f}%)")

# 5. 绘制累积方差解释率折线图
plt.figure(figsize=(8, 5))
plt.plot(range(1, 9), cumulative_variance_ratio, marker='o', linestyle='-', color='b')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Threshold')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(range(1, 9))
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()

# 6. 执行降维 (假设我们选择保留 90% 的方差)
# 从上面的输出和图表中，你可以决定保留多少个成分，通常取累积方差 > 0.90 或 0.95 的 k 值
k = np.argmax(cumulative_variance_ratio >= 0.75) + 1
print(f"\n为了保留至少 90% 的方差，我们需要提取前 {k} 个主成分。")

pca_final = PCA(n_components=k)
X_pca_reduced = pca_final.fit_transform(X_scaled)

print(f"降维后的输入数据矩阵维度: {X_pca_reduced.shape}")