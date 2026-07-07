import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(torch.__version__)

# 数据读取与预处理
# 读取数据
train_df = pd.read_csv("datas/regression/regression_train_data.csv")
test_df = pd.read_csv("datas/regression/regression_test_data.csv")

# 特征列与目标列
feature_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude']
target_col = 'MedHouseVal'

# 提取特征和标签
X = train_df[feature_cols].values
y = train_df[target_col].values.reshape(-1, 1)

# 从训练集中划分出 80% 训练 + 20% 验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_test = test_df[feature_cols].values
y_test = test_df[target_col].values.reshape(-1, 1)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)  # 新增：验证集标准化
X_test_scaled = scaler.transform(X_test)

# 转为 PyTorch Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

print(f"训练集形状: {X_train_tensor.shape}, {y_train_tensor.shape}")
print(f"验证集形状: {X_val_tensor.shape}, {y_val_tensor.shape}")  # 新增：打印验证集形状
print(f"测试集形状: {X_test_tensor.shape}, {y_test_tensor.shape}")


# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel(input_dim=len(feature_cols)).to(device)

# 损失函数与优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降

# 训练模型
epochs = 1000
train_loss_history = []
val_loss_history = []

model.train()
for epoch in range(epochs):
    # 训练阶段
    model.train()
    y_pred = model(X_train_tensor)
    train_loss = criterion(y_pred, y_train_tensor)

    # 反向传播与优化
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # 验证阶段
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_tensor)
        val_loss = criterion(y_val_pred, y_val_tensor)

    # 记录loss
    train_loss_history.append(train_loss.item())
    val_loss_history.append(val_loss.item())

    # 每100轮打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# 测试集评估
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    test_loss = criterion(y_test_pred, y_test_tensor)
    print(f"\n测试集 MSE 损失: {test_loss.item():.4f}")

# 可视化训练+验证损失
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label='Training Loss', color='blue')
plt.plot(val_loss_history, label='Validation Loss', color='red')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.show()

# 输出模型参数
print("\n模型参数（权重与偏置）:")
for name, param in model.named_parameters():
    print(f"{name}: {param}")

# 预测结果示例
print("\n测试集前10条数据预测结果对比:")
print("真实值 | 预测值")
for i in range(10):
    print(f"{y_test[i][0]:.3f} | {y_test_pred[i].item():.3f}")
