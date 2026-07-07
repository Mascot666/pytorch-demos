import os
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np  # 引入 numpy 用于高级像素阈值处理
from torchvision import transforms


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
# 2. 修改后：带自适应二值化清洗的推理函数
# ==========================================
def predict_single_image(image_path, model, transform, device):
    # 1. 加载并转换为灰度图
    img = Image.open(image_path).convert('L')

    # [新增核心优化] 自适应动态二值化提纯（过滤手机拍照阴影）
    # 手机拍照的纸张会有明暗不均。我们通过计算整张图的平均亮度来决定裁切阈值。
    img_np = np.array(img)
    avg_brightness = np.mean(img_np)

    # 动态设定阈值：字迹和背景对比度拉满
    # 拍照时白纸通常较亮。如果整体偏暗，动态下调阈值；整体偏亮，动态上调。
    threshold = int(avg_brightness * 0.82)

    # 将图像中的灰色背景完全变成纯白（255），黑色字迹完全变成纯黑（0），砍掉中间所有的杂色阴影
    img_cleaned = img.point(lambda p: 255 if p > threshold else 0)

    # 2. 自动反色（此时白底黑字会变为完美的【纯黑底、纯白字】，与MNIST格式完全对齐）
    img_inverted = ImageOps.invert(img_cleaned)

    # 3. 缩放到模型要求的 28x28 像素
    img_resized = img_inverted.resize((28, 28))

    # 【调试行】如果你想看一眼被清洗干净后的手写数字图片长什么样，可以解开下面两行代码的注释。
    # 它们会被保存在你当前运行目录的 "cleaned_pics" 文件夹里。
    # os.makedirs("cleaned_pics", exist_ok=True)
    # img_resized.save(f"cleaned_pics/debug_{os.path.basename(image_path)}")

    # 4. 应用标准化的 Tensor 转换
    img_tensor = transform(img_resized).unsqueeze(0).to(device)  # [1, 1, 28, 28]

    # 5. 模型前向传播
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0][predicted.item()].item() * 100

    return predicted.item(), confidence


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"批量预测系统启动，当前使用设备: {device}")

    # 配置模型权重路径和待预测图片文件夹
    WEIGHTS_PATH = 'models/MINST_model_B_weights.pth'
    IMAGE_DIR = 'datas/MNIST_test_digit'
    # 支持的图片格式后缀
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

    # 检查本地参数文件是否存在
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[错误] 未在当前目录下找到 '{WEIGHTS_PATH}' 文件，请先运行训练脚本！")
        exit()

    # 初始化模型并加载权重
    model = NetB().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()  # 切换为测试状态（关闭 Dropout）
    print("本地模型权重加载成功，已启用【自适应阴影滤除技术】，开始扫描文件夹...\n")

    # 定义图像转换流
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 获取文件夹下的所有合法图片
    if not os.path.exists(IMAGE_DIR):
        print(f"[错误] 未找到图片文件夹路径: {IMAGE_DIR}")
        exit()

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if not image_files:
        print(f"[警告] 文件夹 '{IMAGE_DIR}' 内没有找到任何图片文件！")
        exit()

    # 统计字典：用于记录每个数字被预测出来的次数
    results_counter = {i: 0 for i in range(10)}

    print(f"统计到共有 {len(image_files)} 张图片，开始批量识别：")
    print("-" * 60)
    print(f"{'图片文件名':<30} | {'预测数字':<10} | {'置信度(把握)'}")
    print("-" * 60)

    # 循环遍历进行识别
    for img_name in sorted(image_files):
        full_path = os.path.join(IMAGE_DIR, img_name)
        try:
            pred_digit, conf_score = predict_single_image(full_path, model, transform_pipeline, device)
            # 打印单张图片的预测结果
            print(f"{img_name:<30} | {pred_digit:<10} | {conf_score:.2f}%")
            # 累加统计结果
            results_counter[pred_digit] += 1
        except Exception as e:
            print(f"{img_name:<30} | [处理失败] : {str(e)}")

    print("-" * 60)
    print("\n[批量预测完成] 预测数字频次分布统计如下：")
    for digit in range(1, 9):
        print(f"数字 [{digit}] 被识别到了: {results_counter[digit]} 次")
    print(f"数字 [0] 和其他被识别到了: {results_counter[0]} 次")