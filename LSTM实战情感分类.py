import warnings

warnings.filterwarnings("ignore")
import re
import collections
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# ===================== 1. 全局超参数与组件定义 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
EMBED_DIM = 256
HIDDEN_SIZE = 128  # 调大隐藏层维度
NUM_LAYERS = 2  # 增加LSTM层数
VOCAB_MAX_SIZE = 25000  # 扩大词汇表容量
PAD_LEN = 60
EPOCHS = 10
LABEL_MAP = {0: 0, 4: 1}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"@\w+|http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().split()


def build_vocab(token_list, max_vocab):
    word_count = collections.Counter()
    for tokens in token_list:
        word_count.update(tokens)
    common_words = word_count.most_common(max_vocab - 2)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(common_words, start=2):
        vocab[word] = idx
    return vocab


def text2seq(tokens, vocab, max_len):
    seq = [vocab.get(w, 1) for w in tokens]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq


class SentimentDataset(Dataset):
    __slots__ = ["data_x", "data_y"]

    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return torch.tensor(self.data_x[idx]), torch.tensor(self.data_y[idx])


# 改进的模型：双向 LSTM + 全局最大池化
class BiLSTMWithPooling(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        embed = self.embedding(x)
        lstm_out, _ = self.lstm(embed)
        # 通过最大池化打破 Padding 对特征的污染，大幅提升正确率
        pooled_out, _ = torch.max(lstm_out, dim=1)
        out = self.dropout(pooled_out)
        return self.fc(out)


# 训练与评估函数
def train_one_epoch(model, loader, opt, loss_fn, scaler):
    model.train()
    total_loss = 0.0
    for seq, label in loader:
        seq, label = seq.to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast():
            pred = model(seq)
            loss = loss_fn(pred, label)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item() * seq.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    for seq, label in loader:
        seq, label = seq.to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
        with autocast():
            pred = model(seq)
            loss = loss_fn(pred, label)
        total_loss += loss.item() * seq.size(0)
        preds = torch.argmax(pred, dim=1)
        correct += (preds == label).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# ===================== 2. 所有的运行逻辑全部收拢进主入口 =====================
if __name__ == '__main__':
    # 打印环境诊断信息，防止新显卡环境不匹配
    print("====== 环境诊断 ======")
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    print("当前使用的 GPU 设备:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")
    print("======================\n")

    try:
        scaler = GradScaler()

        # 读取数据
        print("正在读取数据...")
        df = pd.read_csv(
            "datas/LSTM情感分类/training.1600000.processed.noemoticon.csv",
            encoding="ISO-8859-1", engine="python", header=None
        )
        df.columns = ["label", "tid", "date", "query", "user", "text"]
        df["sentiment"] = df["label"].map(LABEL_MAP)
        print("标签分布：")
        print(df["sentiment"].value_counts())

        all_texts = df["text"].tolist()
        all_labels = df["sentiment"].tolist()

        print("正在清洗文本...")
        tokenized_texts = [clean_text(txt) for txt in all_texts]

        vocab = build_vocab(tokenized_texts, VOCAB_MAX_SIZE)
        vocab_size = len(vocab)
        print(f"词汇总量：{vocab_size}")

        print("正在转换为序列...")
        all_seqs = np.array([text2seq(tok, vocab, PAD_LEN) for tok in tokenized_texts], dtype=np.int64)
        all_labels = np.array(all_labels, dtype=np.int64)

        # 划分数据集
        train_x, rest_x, train_y, rest_y = train_test_split(
            all_seqs, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        val_x, test_x, val_y, test_y = train_test_split(
            rest_x, rest_y, test_size=0.5, random_state=42, stratify=rest_y
        )
        print(f"训练:{len(train_x)} | 验证:{len(val_x)} | 测试:{len(test_x)}")

        # 数据装载
        train_dataset = SentimentDataset(train_x, train_y)
        val_dataset = SentimentDataset(val_x, val_y)
        test_dataset = SentimentDataset(test_x, test_y)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
                                drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
                                 drop_last=True)

        # 初始化网络
        model = BiLSTMWithPooling(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        train_loss_list, val_loss_list, val_acc_list = [], [], []

        print("\n--- 开始训练 ---")
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)

            scheduler.step()

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            print(
                f"Epoch {epoch:2d}/{EPOCHS} | TrainLoss:{train_loss:.4f} | ValLoss:{val_loss:.4f} | ValAcc:{val_acc:.4f} | LR:{optimizer.param_groups[0]['lr']:.6f}")

        # 测试集最终评估
        test_loss, test_acc = eval_one_epoch(model, test_loader, criterion)
        print(f"\n[最终测试结果] Loss: {test_loss:.4f} | 准确率: {test_acc:.4f}")

        # 绘图曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_list, label="Train Loss")
        plt.plot(val_loss_list, label="Val Loss")
        plt.legend()
        plt.title("Loss Curve")

        plt.subplot(1, 2, 2)
        plt.plot(val_acc_list, c="r", label="Val Accuracy")
        plt.legend()
        plt.title("Val Accuracy")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        traceback.print_exc()