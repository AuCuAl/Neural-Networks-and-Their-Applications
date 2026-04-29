import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# 1. 加载并过滤数据集
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f) # 数据集由json文件组成 [cite: 4]

    poems = []
    for item in data:
        paragraphs = item.get("paragraphs", [])
        poem_text = "".join(paragraphs)

        # 过滤出七言绝句（28个字 + 4个标点符号 = 32个字符）
        if len(poem_text) == 32 and len(paragraphs) == 2:
            poems.append(poem_text)
    return poems

# 这里以提供的 poet.song.40000.json 为例
poems = load_data('poet.song.40000.json')

# 2. 构建词表 (Vocabulary)
chars = set("".join(poems))
# 添加特殊字符
chars.add("<BOS>") # 序列开始标志 (Beginning of Sequence)
chars.add("<EOS>") # 序列结束标志 (End of Sequence)
chars.add("<PAD>") # 填充字符

char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# 3. 创建 PyTorch Dataset
class PoetryDataset(Dataset):
    def __init__(self, poems, char2idx):
        self.data = []
        for poem in poems:
            # 格式: <BOS> 诗句 <EOS>
            encoded = [char2idx["<BOS>"]] + [char2idx[ch] for ch in poem] + [char2idx["<EOS>"]]
            self.data.append(torch.tensor(encoded, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][:-1] # 输入序列
        y = self.data[idx][1:]  # 目标序列（向后平移一位）
        return x, y

dataset = PoetryDataset(poems, char2idx)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PoetryTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    # 生成下三角掩码，防止模型看到未来的字
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x, mask=mask)
        output = self.fc_out(x)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoetryTransformer(vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 20
epoch_losses = []

model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)

        # 调整形状以适应 CrossEntropyLoss: (batch_size * seq_len, vocab_size)
        loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    # 按照作业要求的格式输出 Average Loss [cite: 23]
    print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f} ====')

def generate_poem(model, start_text="明月", max_length=32):
    model.eval()
    # 初始化输入序列，加入 <BOS> 和指定的开篇词
    input_seq = [char2idx["<BOS>"]] + [char2idx[ch] for ch in start_text]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length - len(start_text)):
            output = model(input_tensor)
            # 获取最后一个字符的预测概率分布
            next_token_logits = output[0, -1, :]

            # 使用贪心解码 (Greedy Decoding) 选择概率最大的字
            next_token = torch.argmax(next_token_logits).item()

            if next_token == char2idx["<EOS>"]:
                break

            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]]).to(device)], dim=1)

    generated_indices = input_tensor.squeeze().tolist()
    # 去除 BOS 标志并转换为文本
    poem = "".join([idx2char[idx] for idx in generated_indices[1:]])
    return poem

# 按照作业要求的格式打印演示 [cite: 24]
print("【生成演示】:")
print(generate_poem(model, start_text="明月"))

plt.figure(figsize=(10, 6))
# 绘制 Loss 曲线，样式参考作业图片 [cite: 42, 43]
plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='blue', label='Train Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, epochs + 1))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()