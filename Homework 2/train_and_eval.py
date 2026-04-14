import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import json
import logging
from datetime import datetime

# 导入所有自定义模型
from models import LeNet5, SimpleCNN, MiniAlexNet, ResNetModel, MobileNetModel

# --- 全局变量初始化 ---
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = './dataset' # 指定数据集目录
LOG_DIR = './log'
MODEL_DIR = './model'
HISTORY_DIR = './history'
AUGMENTED_MODEL_DIR = f'{MODEL_DIR}/augmented'
AUGMENTED_HISTORY_DIR = f'{HISTORY_DIR}/augmented'

# 确保必要目录存在
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(AUGMENTED_MODEL_DIR, exist_ok=True)
os.makedirs(AUGMENTED_HISTORY_DIR, exist_ok=True)
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # 解决 OpenMP 冲突，我在最前面导入pandas莫名奇妙解决了OMP: Error #15

# --- 配置 Logging 系统 ---
log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler() # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# --- 数据预处理与加载 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- 训练集数据增强 ---
train_transform_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 SVHN Format 2 数据集 (确保 train_32x32.mat 在 dataset/ 目录下)
train_set = datasets.SVHN(root=DATA_ROOT, split='train', download=True, transform=transform)
train_set_aug = datasets.SVHN(root=DATA_ROOT, split='train', download=True, transform=train_transform_aug)
test_set = datasets.SVHN(root=DATA_ROOT, split='test', download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
train_loader_aug = DataLoader(train_set_aug, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# --- 不增强 ---

def train_and_eval(model):
    model_name = model.model_name
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # 测试阶段
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100. * correct / total

        # 记录数据
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        logger.info(f"[{model_name}] Epoch {epoch+1}/{EPOCHS} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    with open(f'{HISTORY_DIR}/{model_name}.json', 'w') as f:
        json.dump(history, f)
    torch.save(model.state_dict(), f'{MODEL_DIR}/{model_name}.pth')
    logger.info("不增强模型与历史数据 JSON 已保存。")

# --- 增强 ---

def train_and_eval_augmentation(model):
    model_name = model.model_name
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader_aug:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader_aug)
        train_acc = 100. * correct / total

        # 测试阶段
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100. * correct / total

        # 记录数据
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        logger.info(f"[{model_name}] Epoch {epoch+1}/{EPOCHS} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    with open(f'{AUGMENTED_HISTORY_DIR}/{model_name}_augmented.json', 'w') as f:
        json.dump(history, f)
    torch.save(model.state_dict(), f'{AUGMENTED_MODEL_DIR}/{model_name}_.pth')
    logger.info("增强模型与历史数据 JSON 已保存。")

if __name__ == "__main__":
    # 你可以切换选择不同的模型进行测试
    models_to_train = [
        LeNet5(),
        SimpleCNN(),
        MiniAlexNet(),
        ResNetModel(),    # 如果觉得太慢可以注释掉
        MobileNetModel()
    ]

    for model in models_to_train:
        print(f"Starting training with {model.model_name}...")
        train_and_eval(model)
        print("\n")

    model = SimpleCNN()
    print(f"Starting augmentation training with {model.model_name}...")
    train_and_eval_augmentation(model)
    print("\n")
