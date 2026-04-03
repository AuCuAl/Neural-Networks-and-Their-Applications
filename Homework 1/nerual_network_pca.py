import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

# 1. 加载数据
df = pd.read_csv('Concrete_Data_Yeh.csv')
X_raw = df.iloc[:, :8].values
y_raw = df['csMPa'].values.reshape(-1, 1)

# 2. 数据标准化 (PCA 必须步骤)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_raw)

# 3. 执行 PCA 降维
# 设置 n_components=0.90 表示自动选择能解释 90% 方差的主成分数量
pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_scaled)
n_components = X_pca.shape[1]

print(f"原始特征维度: 8")
print(f"PCA 降维后维度 (解释90%方差): {n_components}")
print(f"各主成分方差贡献率: {pca.explained_variance_ratio_}")

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_raw, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# 5. 构建针对 PCA 输入的神经网络
class PCANet(nn.Module):
    def __init__(self, input_dim):
        super(PCANet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), # 动态设置输入维度为 n_components
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = PCANet(n_components)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. 模型训练
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 7. 模型评估与预测
model.eval()
with torch.no_grad():
    y_pred_t = model(X_test_t)
    y_pred = y_pred_t.numpy()
    test_mse = criterion(y_pred_t, y_test_t)

print(f"\n[Neural Network] Final Test MSE: {test_mse.item():.4f}")
# e) 绘图：将测试集的 target 与模型的 output 进行可视化比较
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predictions')
# 散点图
# 细节将 tensor 转为 numpy 类型方便绘图

# 绘制 y=x 参考线 (完美预测线)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.title('Neural Network Performance: Actual vs Predicted Output')
plt.xlabel('Actual Concrete Compressive Strength (MPa)')
plt.ylabel('Predicted Output (MPa)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()

# 8. 自动布局绘制主成分与预测值的关系图
def plot_pca_results(X_pca_test, y_actual, y_predicted):
    n_feat = X_pca_test.shape[1]
    n_rows = 2
    n_cols = math.ceil(n_feat / n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2 * n_cols))
    axes_flat = axes.flatten() if n_feat > 1 else [axes]

    for i in range(n_feat):
        ax = axes_flat[i]
        pc_data = X_pca_test[:, i]

        ax.scatter(pc_data, y_actual, color='#1f77b4', alpha=0.5, label='Actual', marker='o')
        ax.scatter(pc_data, y_predicted, color='#d62728', alpha=0.6, label='Predicted', marker='x')

        # 设定 X 轴范围
        x_min, x_max = pc_data.min(), pc_data.max()
        margin = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - margin, x_max + margin)

        ax.set_title(f'Strength vs Principal Component {i+1}')
        ax.set_xlabel(f'PC{i+1} Score')
        ax.set_ylabel('Strength (MPa)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)

    # 关闭多余的子图
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()

plot_pca_results(X_test, y_test, y_pred)