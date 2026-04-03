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
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)
# 似乎是一个内置函数，用于将按比例分割，不过似乎并不是前80%后20%的分割方法
# 查询后得知，random_state 参数是一个随机数种子，用于在分割前对数据进行洗牌，保证结果的可重复性
# shuffle 参数决定了是否在分割前对数据进行洗牌，默认为True，即进行洗牌
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# 将shuffle置否即可实现前80%后20%的分割方法

# b) 数据预处理：使用标准化处理特征，确保输入数据在相同的量纲维度下，加速模型收敛
scaler = StandardScaler()
# StandardScaler 是一种常用的数据标准化方法，用于将数据转换为均值为 0，标准差为 1 的标准正态分布
# 均值为 0，标准差为 1 正确，但是否为正态分布未知
X_train_scaled = scaler.fit_transform(X_train)
# fit_transform 不仅计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正态分布
X_test_scaled = scaler.transform(X_test)
# transform 很显然，它只是进行转换，只是把测试数据转换成标准的正态分布。在fit的基础上，进行标准化，降维，归一化等操作
# fit 简单来说，就是求得训练集X的均值，方差，最大值，最小值,这些训练集X固有的属性
# 一般来说测试集使用fit_transform验证集使用transform
# 在测试集时已经收集了数据的均值和方差并且我们认为测试集和验证集数据收集的都足够充分，均值和方差一致，所以验证集不需要重新求均值和方差
# 也就是说transform是直接拿fit_transform训练好的参数去标准化别的数据集？原来标准化处理的时候就已经有训练了
# 自变量X进行了标准化处理，但因变量y并没有

# 将 NumPy 数组转换为 PyTorch 的 Tensor 格式
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)
# 经典pytorch，明显是要上神经网络了
# 我用了gpu版的2.10.0+cu126的pytorch

# ==========================================
# 步骤 3: 神经网络构建
# ==========================================

# c) 构建适合回归问题的神经网络模型，包含输入层(8)、多个隐藏层和输出层(1)
class ConcreteNet(nn.Module):
    def __init__(self):
        super(ConcreteNet, self).__init__()
        # 使用多个隐藏层和 ReLU 激活函数测试效果
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # 回归问题输出层不需要激活函数
        )
        # 8-64-32-1共四层，64后、32后加ReLU

    def forward(self, x): # 前向传播
        return self.net(x)

model = ConcreteNet()
# 实例化ConcreteNet
# 使用均方误差 (MSE) 作为损失函数
criterion = nn.MSELoss()
# 使用 Adam 优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)
# lr是精度？

# ==========================================
# 步骤 4: 模型训练
# ==========================================

# d) 使用训练集对模型进行训练，通过反向传播算法更新权重和偏置
epochs = 500
# 当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一次epoch
train_losses = []

for epoch in range(epochs):
    model.train()
    # 开始训练模型？
    optimizer.zero_grad()           # 清空梯度

    outputs = model(X_train_t)      # 前向传播
    loss = criterion(outputs, y_train_t) # 计算损失

    loss.backward()                 # 反向传播
    optimizer.step()                # 更新权重参数

    train_losses.append(loss.item())
    # item()函数的主要用途是从单元素张量中提取元素值并返回，同时保持元素的类型不变
    # 看来 criterion 返回的是张量

    if (epoch + 1) % 100 == 0:
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

plot(X_test_t, y_test_t, predictions, df.columns[:8].tolist())