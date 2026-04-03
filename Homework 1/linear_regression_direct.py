import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plot import plot

# ==========================================
# 步骤 1 & 2: 数据准备与预处理
# ==========================================

# 读取数据集
df = pd.read_csv('Concrete_Data_Yeh.csv')

# 提取8个输入变量和1个输出变量
# iloc方法是通过索引行、列的索引位置[index, columns]来寻找值
# 第一个参数为行，第二个参数为列
# csv文件的第一行为标签，不计入，df.iloc[0]即表示第2行的数据
X = df.iloc[:, :8].values  # 前8列为输入特征
y = df.iloc[:, 8].values.reshape(-1, 1)  # 第9列为输出特征 (抗压强度)
# 这里不加.values好像也行，加了返回array
# reshape把行向量转变为列向量，相当于转置
# 这是因为iloc[:, 8]破坏了列维度，使用iloc[:, 8:9]或iloc[:, 8:]即可保留列维度，不需要reshape

# a) 数据准备：将数据集的前80%作为训练集，后20%作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# 将shuffle置否即可实现前80%后20%的分割方法

# b) 数据预处理：使用标准化处理特征，确保输入数据在相同的量纲维度下，加速模型收敛
scaler = StandardScaler()
# StandardScaler 是一种常用的数据标准化方法，用于将数据转换为均值为 0，标准差为 1 的标准正态分布
X_train_scaled = scaler.fit_transform(X_train)
# fit_transform 不仅计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据
X_test_scaled = scaler.transform(X_test)
# transform 很显然，它只是进行转换，只是把测试数据转换成标准的正态分布。在fit的基础上，进行标准化
# 一般来说测试集使用fit_transform验证集使用transform
# 在测试集时已经收集了数据的均值和方差并且我们认为测试集和验证集数据收集的都足够充分，均值和方差一致，所以验证集不需要重新求均值和方差

# 将 NumPy 数组转换为 PyTorch 的 Tensor 格式
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)
# 经典pytorch，明显是要上模型了

# ==========================================
# 步骤 3: 线性回归模型构建 (核心修改部分)
# ==========================================

# c) 构建线性回归模型
class LinearRegressionNet(nn.Module):
    def __init__(self):
        super(LinearRegressionNet, self).__init__()
        # 线性回归不需要隐藏层和激活函数，直接从输入层(8)映射到输出层(1)
        # 这就等同于 y = w1*x1 + w2*x2 + ... + w8*x8 + b
        self.linear = nn.Linear(8, 1)

    def forward(self, x): # 前向传播
        return self.linear(x)

model = LinearRegressionNet()
# 实例化 LinearRegressionNet

# 使用均方误差 (MSE) 作为损失函数
criterion = nn.MSELoss()

# 使用 Adam 优化器 (对于标准线性回归也可以用 optim.SGD 随机梯度下降)
# lr (learning rate) 是学习率，决定每次参数更新的步长大小
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 步骤 4: 模型训练
# ==========================================

# d) 使用训练集对模型进行训练，通过反向传播算法更新权重和偏置
epochs = 5000
# 当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一次epoch
train_losses = []

for epoch in range(epochs):
    model.train()
    # 开始训练模型
    optimizer.zero_grad()           # 清空梯度

    outputs = model(X_train_t)      # 前向传播
    loss = criterion(outputs, y_train_t) # 计算损失

    loss.backward()                 # 反向传播，计算当前梯度
    optimizer.step()                # 根据梯度更新权重参数 (w 和 b)

    train_losses.append(loss.item())
    # item()函数的主要用途是从单元素张量中提取元素值并返回

    if (epoch + 1) % 1000 == 0:
        # 每 100 epoch
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# ==========================================
# 步骤 5: 模型测试、评估与可视化
# ==========================================

# e) 使用测试集评估模型，计算预测值与实际值之间的均方误差
model.eval() # 切换至评估模式
with torch.no_grad():
    # torch.no_grad 是 PyTorch 中的一个上下文管理器，它的主要作用是在某个指定的代码块中禁用梯度计算。
    # 这在进行模型推理或评估时非常有用，因为在这些情况下，我们通常不需要计算梯度，从而可以减少内存消耗和计算时间
    predictions = model(X_test_t)
    test_mse = criterion(predictions, y_test_t)

print(f"\nFinal Test Mean Squared Error (MSE): {test_mse.item():.4f}")

# 调用外部 plot 函数进行绘图
plot(X_test_t, y_test_t, predictions, df.columns[:8].tolist())