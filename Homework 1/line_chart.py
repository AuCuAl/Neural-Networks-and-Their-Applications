import numpy as np
import matplotlib.pyplot as plt

# 确保此时你有以下变量：
# X_test: 原始未标准化的测试集特征矩阵 (为了横坐标有实际物理意义)
# y_test: 真实的测试集输出
# y_pred: 模型对测试集的预测输出
# selected_features: ['cement', 'superplasticizer', 'age', 'water']

X_test = X_test_t
y_test = y_test_t
y_pred = predictions

# 创建一个 2x2 的画布，共 4 张子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten() # 将二维的轴矩阵展平，方便使用 for 循环遍历

for i, feature in enumerate(selected_features):
    ax = axes[i]

    # 获取当前特征在测试集中的原始数值
    x_feature_raw = X_test[:, i]

    # 【关键步骤】按当前特征的数值从小到大进行排序，防止画出的线乱飞
    sort_idx = np.argsort(x_feature_raw)
    x_sorted = x_feature_raw[sort_idx]
    y_test_sorted = y_test[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    # 绘制真实值 (蓝色实心圆点 + 实线)
    ax.plot(x_sorted, y_test_sorted, marker='o', linestyle='-', color='#1f77b4',
            alpha=0.6, label='Actual Strength', markersize=5, linewidth=1.5)

    # 绘制预测值 (红色方块 + 虚线)
    ax.plot(x_sorted, y_pred_sorted, marker='s', linestyle='--', color='#d62728',
            alpha=0.6, label='Predicted Strength', markersize=5, linewidth=1.5)

    # 设置图表元素
    ax.set_title(f'Strength vs {feature.capitalize()}')
    ax.set_xlabel(f'{feature} (Original Scale)')
    ax.set_ylabel('Compressive Strength (MPa)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)

# 自动调整子图间距，防止文字重叠
plt.tight_layout()
plt.show()