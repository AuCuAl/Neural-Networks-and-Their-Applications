import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # 导入 PCA 模块
from plot import plot

# ==========================================
# 步骤 1 & 2: 数据准备、标准化与 PCA 降维
# ==========================================

# 读取数据集
df = pd.read_csv('Concrete_Data_Yeh.csv')

X = df.iloc[:, :8].values  # 原始 8 个特征
y = df.iloc[:, 8].values.reshape(-1, 1)

# a) 数据分割：先分割，再在训练集上寻找主成分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# b) 数据预处理：必须先进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 核心修改：PCA 降维 ---
# 选择前 6 个主成分
n_components = 6
pca = PCA(n_components=n_components)

# 在训练集上拟合 PCA 并在训练/测试集上应用转换
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"原始特征维度: {X_train_scaled.shape[1]}")
print(f"PCA 降维后维度: {X_train_pca.shape[1]}")
print(f"前 {n_components} 个主成分解释的总方差比例: {sum(pca.explained_variance_ratio_):.4f}")

# 将处理后的数据转换为 Tensor
X_train_t = torch.tensor(X_train_pca, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test_pca, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# ==========================================
# 步骤 3: 线性回归模型构建 (输入维度改为 6)
# ==========================================

class LinearRegressionNet(nn.Module):
    def __init__(self):
        super(LinearRegressionNet, self).__init__()
        # 现在的输入维度是 PCA 处理后的 6 个分量
        self.linear = nn.Linear(n_components, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 步骤 4: 模型训练
# ==========================================

epochs = 5000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# ==========================================
# 步骤 5: 模型评估与可视化
# ==========================================

model.eval()
with torch.no_grad():
    predictions = model(X_test_t)
    test_mse = criterion(predictions, y_test_t)

print(f"\nFinal Test Mean Squared Error (MSE) with 6 PCA Components: {test_mse.item():.4f}")

# 注意：由于输入变为了 PCA 主成分，原始的特征列名不再适用
# 传入新的标签 [PC1, PC2, ..., PC6]
pca_labels = [f'PC{i+1}' for i in range(n_components)]
plot(X_test_t, y_test_t, predictions, pca_labels)