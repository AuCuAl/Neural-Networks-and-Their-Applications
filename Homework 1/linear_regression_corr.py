import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plot import plot

# ==========================================
# 步骤 1 & 2: 数据准备与预处理 (基于筛选特征)
# ==========================================

# 读取数据集
df = pd.read_csv('Concrete_Data_Yeh.csv')

# --- 修改点：基于相关性分析筛选的特征列表 ---
selected_features = ['cement', 'superplasticizer', 'age', 'water']
print(f"使用的主要特征: {selected_features}")

# 提取筛选后的输入变量 X 和 输出变量 y
# 使用 df[selected_features] 直接按列名提取
X = df[selected_features].values
y = df.iloc[:, 8].values.reshape(-1, 1)  # 假设抗压强度依然在第9列

# a) 数据准备：前80%作为训练集，后20%作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# b) 数据预处理：标准化
# 此时 scaler 仅针对这 4 个特征进行计算
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将 NumPy 数组转换为 PyTorch 的 Tensor 格式
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# ==========================================
# 步骤 3: 线性回归模型构建 (输入维度改为 4)
# ==========================================

class LinearRegressionNet(nn.Module):
    def __init__(self):
        super(LinearRegressionNet, self).__init__()
        # 注意：这里的输入维度从 8 变成了 4
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 步骤 4: 模型训练 (代码逻辑不变)
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

print(f"\nFinal Test Mean Squared Error (MSE) with 4 features: {test_mse.item():.4f}")

# 可视化时，特征名称列表传入筛选后的列表
plot(X_test_t, y_test_t, predictions, selected_features)