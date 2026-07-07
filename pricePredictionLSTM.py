import math
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置 CPU 或 GPU 加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_file = "datas/LSTM股票预测/DIS.csv"


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# fix random seed for reproducibility
np.random.seed(5)
torch.manual_seed(5)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(5)

# load the dataset
df = read_csv(input_file, header=None, index_col=None, delimiter=",")

# take close price column[5]
all_y = df[5].values
dataset = all_y.reshape(-1, 1)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets, 50% test data, 50% training data
train_size = int(len(dataset) * 0.5)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size : len(dataset), :]

# reshape into X=t and Y=t+1, timestep 240
look_back = 240
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# 转换为 PyTorch Tensors
X_train_t = torch.tensor(trainX, dtype=torch.float32).to(device)
y_train_t = torch.tensor(trainY, dtype=torch.float32).reshape(-1, 1).to(device)
X_test_t = torch.tensor(testX, dtype=torch.float32).to(device)
y_test_t = torch.tensor(testY, dtype=torch.float32).reshape(-1, 1).to(device)

# 创建 DataLoader 以满足 batch_size=240 的训练需求
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=240, shuffle=True)


# 定义 PyTorch LSTM 模型
class LSTMNet(nn.Module):
    def __init__(
        self, input_size=1, hidden_size=25, output_size=1, dropout_prob=0.1
    ):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


# 实例化模型
model = LSTMNet(
    input_size=1, hidden_size=25, output_size=1, dropout_prob=0.1
).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型 (1000 epochs)
epochs = 1000
model.train()
train_loss_list = []  # 保存每轮epoch损失，用于画图

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)

    total_loss = epoch_loss / len(train_dataset)
    train_loss_list.append(total_loss)  # 存入损失列表
    # 打印训练进度 (每 100 轮打印一次)
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.6f}")

# 切换到评估模式进行预测
model.eval()
with torch.no_grad():
    trainPredict = model(X_train_t).cpu().numpy()
    testPredict = model(X_test_t).cpu().numpy()

# 把真实值转回原本的 numpy 格式以便反归一化
trainY = trainY.reshape(1, -1)
testY = testY.reshape(1, -1)

# 反归一化还原真实价格
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

# 计算并打印均方根误差 (RMSE)
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print("Train Score: %.2f RMSE" % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print("Test Score: %.2f RMSE" % (testScore))

# 移动训练集的预测结果位置以供绘图
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back : len(trainPredict) + look_back, :] = trainPredict

# 移动测试集的预测结果位置以供绘图
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[
    len(trainPredict) + (look_back * 2) + 1 : len(dataset) - 1, :
] = testPredict

# 打印控制台输出
print("testPrices:")
testPrices = scaler.inverse_transform(dataset[test_size + look_back :])
print(testPrices)

print("testPredictions:")
print(testPredict)

# ========== 图1：股价真实值+训练预测+测试预测 ==========
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(dataset), label="Actual Price", color="blue")
plt.plot(trainPredictPlot, label="Train Predict", color="green")
plt.plot(testPredictPlot, label="Test Predict", color="red")
plt.title("Disney Stock Price Prediction (LSTM)")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.show()

# ========== 图2：新增 RMSE/LOSS 迭代曲线图 ==========
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_loss_list, color='darkred', linewidth=1.5)
plt.title('LSTM Training Loss Curve (MSE Loss)')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss Value')
plt.grid(True, alpha=0.3)
plt.show()