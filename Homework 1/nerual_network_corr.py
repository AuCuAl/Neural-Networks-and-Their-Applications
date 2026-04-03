import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plot import plot

# ==========================================
# 步骤 1 & 2: 数据准备与特征选择
# ==========================================
df = pd.read_csv('Concrete_Data_Yeh.csv')

# 只选择相关性大于 0.2 的四个特征
selected_features = ['cement', 'superplasticizer', 'age', 'water']
X = df[selected_features].values
y = df['csMPa'].values.reshape(-1, 1)

# 数据集划分与标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为 PyTorch Tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# ==========================================
# 步骤 3: 神经网络构建 (针对4维输入进行修改)
# ==========================================
class ConcreteNetSelected(nn.Module):
    def __init__(self):
        super(ConcreteNetSelected, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),  # 【关键修改】输入层维度改为 4
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ConcreteNetSelected()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 步骤 4 & 5: 模型训练与测试
# ==========================================
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(X_test_t)
    test_mse = criterion(predictions, y_test_t)

print(f"\n[Neural Network] Final Test MSE: {test_mse.item():.4f}")

plot(X_test_t, y_test_t, predictions, selected_features)