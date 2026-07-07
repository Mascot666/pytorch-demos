import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ----------------------------------------
# 1. 自定义 CIFAR-10 本地数据集加载器
# ----------------------------------------
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class LocalCIFAR10Dataset(Dataset):
    def __init__(self, data_dir, train=True):
        self.data = []
        self.targets = []
        if train:
            for i in range(1, 6):
                file_path = os.path.join(data_dir, f'data_batch_{i}')
                batch_dict = unpickle(file_path)
                self.data.append(batch_dict[b'data'])
                self.targets.extend(batch_dict[b'labels'])
            self.data = np.concatenate(self.data)
        else:
            file_path = os.path.join(data_dir, 'test_batch')
            batch_dict = unpickle(file_path)
            self.data = batch_dict[b'data']
            self.targets = batch_dict[b'labels']

        self.data = self.data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.targets[idx], dtype=torch.long)


# ----------------------------------------
# 2. 三种不同的网络结构定义
# ----------------------------------------
class FlexibleCNN(nn.Module):
    def __init__(self, config_type='model_A'):
        super(FlexibleCNN, self).__init__()
        self.config_type = config_type

        if config_type == 'model_A':
            # 结构 1：浅层网络，较少卷积核（易发生欠拟合）
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )

        elif config_type == 'model_B':
            # 结构 2：标准卷积网络（适度扩展）
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 8 * 8, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )

        elif config_type == 'model_C':
            # 结构 3：深层复杂网络，大卷积核 + 多通道（在没有Dropout时易发生过拟合）
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------------------
# 3. 单个模型的训练核心逻辑
# ----------------------------------------
def train_model(model_config, train_loader, test_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlexibleCNN(config_type=model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        # 训练
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 验证
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['train_acc'].append(100.0 * correct / total)
        history['test_loss'].append(test_loss / len(test_loader.dataset))
        history['test_acc'].append(100.0 * test_correct / test_total)

        print(
            f"[{model_config}] Epoch {epoch + 1:02d} | Train Acc: {history['train_acc'][-1]:.1f}% | Test Acc: {history['test_acc'][-1]:.1f}%")

    return history


# ----------------------------------------
# 4. 主程序：循环训练三个模型并作图对比
# ----------------------------------------
if __name__ == '__main__':
    DATA_DIR = 'datas/cifar-10-batches-py' 
    EPOCHS = 10
    BATCH_SIZE = 128

    # 准备数据流
    train_dataset = LocalCIFAR10Dataset(DATA_DIR, train=True)
    test_dataset = LocalCIFAR10Dataset(DATA_DIR, train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 存储三个模型的结果
    all_results = {}
    models_to_test = ['model_A', 'model_B', 'model_C']

    for model_name in models_to_test:
        print(f"\n>>> 正在训练 {model_name} ...")
        all_results[model_name] = train_model(model_name, train_loader, test_loader, epochs=EPOCHS)

    # --- 开始绘制多模型对比图 ---
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(14, 10))

    # 颜色配置，确保对比清晰
    colors = {'model_A': 'g', 'model_B': 'b', 'model_C': 'r'}

    # 1. 训练与测试 Loss 对比
    plt.subplot(2, 2, 1)
    for name in models_to_test:
        plt.plot(epochs_range, all_results[name]['train_loss'], color=colors[name], linestyle='--',
                 label=f'{name} (Train)')
        plt.plot(epochs_range, all_results[name]['test_loss'], color=colors[name], linestyle='-', marker='o',
                 label=f'{name} (Test)')
    plt.title('Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. 训练与测试 准确率 对比
    plt.subplot(2, 2, 2)
    for name in models_to_test:
        plt.plot(epochs_range, all_results[name]['train_acc'], color=colors[name], linestyle='--',
                 label=f'{name} (Train)')
        plt.plot(epochs_range, all_results[name]['test_acc'], color=colors[name], linestyle='-', marker='s',
                 label=f'{name} (Test)')
    plt.title('Accuracy Comparison (%)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # 3. 专门提取测试集准确率进行直观性能对比
    plt.subplot(2, 2, 3)
    for name in models_to_test:
        plt.plot(epochs_range, all_results[name]['test_acc'], color=colors[name], marker='v', linewidth=2,
                 label=f'{name} Test Acc')
    plt.title('Pure Test Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()