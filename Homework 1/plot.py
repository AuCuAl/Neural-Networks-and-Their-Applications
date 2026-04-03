import math
import matplotlib.pyplot as plt

# 确保此时你有以下变量：
# X_test: 原始未标准化的测试集特征矩阵 (为了横坐标有实际物理意义)
# y_test: 真实的测试集输出
# y_pred: 模型对测试集的预测输出
# selected_features: ['cement', 'superplasticizer', 'age', 'water']

def plot(X_test, y_test, y_pred, selected_features):
    # e) 绘图：将测试集的 target 与模型的 output 进行可视化比较
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred.numpy(), alpha=0.6, color='blue', label='Predictions')
    # 散点图
    # 细节将 tensor 转为 numpy 类型方便绘图

    # 绘制 y=x 参考线 (完美预测线)
    min_val = min(y_test.min(), y_pred.numpy().min())
    max_val = max(y_test.max(), y_pred.numpy().max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.title('Neural Network Performance: Actual vs Predicted Output')
    plt.xlabel('Actual Concrete Compressive Strength (MPa)')
    plt.ylabel('Predicted Output (MPa)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

    n_features = len(selected_features)
    n_rows = 2  # 你可以根据喜好设置为 2 或 3
    n_cols = math.ceil(n_features / n_rows)

    # 创建一个 2x2 的画布，共 4 张子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    axes = axes.flatten() # 将二维的轴矩阵展平，方便遍历

    for i, feature in enumerate(selected_features):
        ax = axes[i]

        # 获取当前特征在测试集中的原始数值
        # 散点图不需要对数据进行排序，直接映射即可
        x_feature_raw = X_test[:, i]

        # 绘制真实值 (蓝色圆点，s为点的大小)
        ax.scatter(x_feature_raw, y_test, color='#1f77b4', alpha=0.6, label='Actual Strength', marker='o', s=40)

        # 绘制预测值 (红色叉号，使用不同形状避免颜色重叠时看不清)
        ax.scatter(x_feature_raw, y_pred, color='#d62728', alpha=0.7, label='Predicted Strength', marker='x', s=40)

        # 设置图表元素
        ax.set_title(f'Strength vs {feature.capitalize()}')
        ax.set_xlabel(f'{feature} (Original Scale)')
        ax.set_ylabel('Compressive Strength (MPa)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)

        x_min, x_max = x_feature_raw.min(), x_feature_raw.max()
        x_margin = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

    # 自动调整子图间距
    plt.tight_layout()
    plt.show()