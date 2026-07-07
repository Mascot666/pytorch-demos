import os
import re
import traceback
import warnings

# 屏蔽所有类型的警告与提示
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # 仅显示致命错误，屏蔽提示

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ===================== 2. 全局超参数与组件定义 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024  # 16G 显存
PAD_LEN = 60  # 推文截断长度
EPOCHS = 4  # BERT 标准微调轮数
LR = 4e-5  # 配合 1024 大 Batch 最佳学习率
LABEL_MAP = {0: 0, 4: 1}

# 选用轻量高效的 DistilBERT
MODEL_NAME = "distilbert-base-uncased"


def clean_text_for_bert(text):
    text = str(text).lower()
    text = re.sub(r"@\w+|http\S+", "", text)
    return text.strip()


class BertSentimentDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# 训练函数
def train_one_epoch(model, loader, opt, scheduler, scaler, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(
        total=len(loader), desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False
    )

    for batch in loader:
        opt.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        current_loss = loss.item()
        total_loss += current_loss * input_ids.size(0)

        pbar.set_postfix({"Loss": f"{current_loss:.4f}"})
        pbar.update(1)

    pbar.close()
    return total_loss / len(loader.dataset)


# 评估函数
@torch.no_grad()
def eval_one_epoch(model, loader, desc="[Val]"):
    model.eval()
    total_loss = 0.0
    correct = 0

    pbar = tqdm(total=len(loader), desc=desc, leave=False)

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()

        pbar.update(1)

    pbar.close()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# ===================== 3. 主入口 =====================
if __name__ == "__main__":
    print("====== 环境诊断 ======")
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    print("======================\n")

    try:
        scaler = GradScaler()

        # 1. 读取数据
        print("正在读取数据...")
        df = pd.read_csv(
            "datas/LSTM情感分类/training.1600000.processed.noemoticon.csv",
            encoding="ISO-8859-1",
            engine="python",
            header=None,
        )
        df.columns = ["label", "tid", "date", "query", "user", "text"]

        # 映射标签并剔除不在映射范围内的异常行
        df["sentiment"] = df["label"].map(LABEL_MAP)
        df = df.dropna(subset=["sentiment"])

        # 清洗文本
        print("正在清洗文本...")
        df["text"] = df["text"].fillna("")
        df["cleaned_text"] = df["text"].apply(clean_text_for_bert)

        df = df[df["cleaned_text"].str.strip() != ""]

        all_texts = df["cleaned_text"].tolist()
        all_labels = df["sentiment"].astype(int).tolist()

        # 2. 划分数据集
        train_x, rest_x, train_y, rest_y = train_test_split(
            all_texts,
            all_labels,
            test_size=0.2,
            random_state=42,
            stratify=all_labels,
        )
        val_x, test_x, val_y, test_y = train_test_split(
            rest_x, rest_y, test_size=0.5, random_state=42, stratify=rest_y
        )

        # 强制转换为纯 Python List
        train_x, val_x, test_x = list(train_x), list(val_x), list(test_x)
        train_y, val_y, test_y = list(train_y), list(val_y), list(test_y)

        print(
            f"训练集: {len(train_x)} | 验证集: {len(val_x)} | 测试集: {len(test_x)}"
        )

        # 3. 初始化分词器
        print(f"正在加载 {MODEL_NAME} 分词器...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


        # 4. 纯列表分词辅助函数
        def batch_tokenize_pure(texts, tokenizer, max_len, name, chunk_size=100000):
            all_input_ids = []
            all_attention_mask = []

            total_chunks = (len(texts) + chunk_size - 1) // chunk_size
            pbar = tqdm(total=total_chunks, desc=f"分词进度 [{name}]")

            for i in range(0, len(texts), chunk_size):
                chunk = texts[i: i + chunk_size]
                encoded = tokenizer(
                    chunk, truncation=True, padding="max_length", max_length=max_len
                )
                all_input_ids.extend(encoded["input_ids"])
                all_attention_mask.extend(encoded["attention_mask"])
                pbar.update(1)

            pbar.close()
            print(f"正在将 [{name}] 转换为全局 PyTorch 张量...")
            return {
                "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(
                    all_attention_mask, dtype=torch.long
                ),
            }


        print("\n正在进行全局批量分词...")
        train_encodings = batch_tokenize_pure(
            train_x, tokenizer, PAD_LEN, "Train"
        )
        val_encodings = batch_tokenize_pure(val_x, tokenizer, PAD_LEN, "Val")
        test_encodings = batch_tokenize_pure(test_x, tokenizer, PAD_LEN, "Test")
        print("全局分词与张量化完成！数据已安全就绪。\n")

        # 5. 实例化 Dataset & DataLoader
        train_dataset = BertSentimentDataset(train_encodings, train_y)
        val_dataset = BertSentimentDataset(val_encodings, val_y)
        test_dataset = BertSentimentDataset(test_encodings, test_y)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # 6. 加载预训练模型
        print(f"正在加载 {MODEL_NAME} 预训练模型...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        ).to(DEVICE)

        # 7. 优化器与 Warmup 调度器
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)

        total_steps = len(train_loader) * EPOCHS
        num_warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )

        train_loss_list, val_loss_list, val_acc_list = [], [], []

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, epoch
            )

            val_loss, val_acc = eval_one_epoch(
                model, val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"
            )

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            print(
                f"Epoch {epoch:2d}/{EPOCHS} | "
                f"TrainLoss:{train_loss:.4f} | "
                f"ValLoss:{val_loss:.4f} | "
                f"ValAcc:{val_acc:.4f} | "
                f"LR:{optimizer.param_groups[0]['lr']:.2e}"
            )

        # 8. 测试集最终评估
        test_loss, test_acc = eval_one_epoch(
            model, test_loader, desc="[Final Test]"
        )
        print(
            f"\n[BERT最终测试结果] Loss: {test_loss:.4f} | 准确率: {test_acc:.4f}"
        )

        # 9. 绘图
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_list, label="Train Loss")
        plt.plot(val_loss_list, label="Val Loss")
        plt.legend()
        plt.title("BERT Loss Curve")

        plt.subplot(1, 2, 2)
        plt.plot(val_acc_list, c="r", label="Val Accuracy")
        plt.legend()
        plt.title("BERT Val Accuracy")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        traceback.print_exc()
