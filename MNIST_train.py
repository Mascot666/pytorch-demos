import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 0.001

# 数据预处理
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 确保是单通道灰度图
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# ==========================================
# 2. 定义两种不同结构的卷积神经网络
# ==========================================

# 结构 A：浅层网络
class NetA(nn.Module):
    def __init__(self):
        super(NetA, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 结构 B：深层网络
class NetB(nn.Module):
    def __init__(self):
        super(NetB, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==========================================
# 3. 训练与测试通用函数
# ==========================================
def train_and_evaluate(model, model_name, train_loader, test_loader):
    print(f"\n--- 开始训练网络: {model_name} ---")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # 每一个 epoch 结束后进行测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch}/{EPOCHS}] - 训练 Loss: {epoch_loss:.4f} - 测试准确率: {accuracy:.2f}%")

    return model


# ==========================================
# 4. 主程序入口 (Windows 环境安全避坑)
# ==========================================
if __name__ == '__main__':
    print(f"主进程启动，正在使用设备: {device}")

    # 请根据实际情况修改数据集文件夹路径
    TRAIN_DIR = "datas/MNIST - JPG - testing"
    TEST_DIR = "datas/MNIST - JPG - training"


    # 加载数据集
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 训练网络 A 和网络 B
    model_A = train_and_evaluate(NetA(), "结构 A (浅层/3x3核)", train_loader, test_loader)
    model_B = train_and_evaluate(NetB(), "结构 B (深层/5x5核)", train_loader, test_loader)

    # 保存训练好的参数到本地硬盘
    torch.save(model_A.state_dict(), 'models/MINST_model_A_weights.pth')
    torch.save(model_B.state_dict(), 'models/MINST_model_B_weights.pth')
    print("\n[系统提示] 两个模型的权重已成功保存到models ('MINST_model_A_weights.pth', 'MINST_model_B_weights.pth')！")