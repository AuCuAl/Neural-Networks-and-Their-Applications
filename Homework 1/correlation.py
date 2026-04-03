import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 读取数据集
df = pd.read_csv('D:\\Microelectronics\\neural-networks-and-their-applications\\Homework 1\\Concrete_Data_Yeh.csv')

# 2. 计算所有变量之间的皮尔逊相关系数矩阵
correlation_matrix = df.corr()

# 3. 单独提取各特征与输出变量 (csMPa) 的相关性
# drop('csMPa') 是为了把目标变量自己和自己的相关性(1.0)去掉
target_corr = correlation_matrix['csMPa'].drop('csMPa')

# 按照相关性的绝对值从大到小排序，方便寻找最重要的特征
target_corr_sorted = target_corr.abs().sort_values(ascending=False)

print("=== 各特征与混凝土抗压强度(csMPa)的绝对相关性排序 ===")
for feature, corr_value in target_corr_sorted.items():
    # 获取原始的正负号相关性值
    original_val = target_corr[feature]
    print(f"{feature:20s}: 绝对值 {corr_value:.4f} (原始值: {original_val:.4f})")

# 4. 绘制相关性热力图可视化全貌
plt.figure(figsize=(10, 8)).canvas.manager.set_window_title('Figure 2')
# annot=True 显示具体数值, cmap='coolwarm' 提供红蓝冷暖色调对比
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Concrete Features')
plt.tight_layout()
plt.show()

# 5. 特征选择（示例：选择绝对相关性大于 0.1 的特征）
threshold = 0.2
selected_features = target_corr_sorted[target_corr_sorted > threshold].index.tolist()
print(f"\n基于阈值 {threshold} 选择的主要特征: {selected_features}")