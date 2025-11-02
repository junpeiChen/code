import os
import sys
import logging
import time

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 设置离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 强制使用GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 如果使用GPU，显示GPU信息
if device.type == 'cuda':
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    # 设置CUDA设备（如果有多个GPU）
    torch.cuda.set_device(0)

# 定义本地模型路径
#local_model_path = r"D:\Pycharm\bert-base-uncased"
#服务器下地址
local_model_path = r"/root/autodl-tmp/bert-base-uncased"

# 检查本地模型是否存在
if not os.path.exists(local_model_path):
    print(f"错误: 本地模型路径不存在: {local_model_path}")
    print("请确保你已经下载了 bert-base-uncased 模型到该路径")
    sys.exit(1)

try:
    # 从本地加载tokenizer和模型，并直接放到GPU上
    tokenizer = BertTokenizerFast.from_pretrained(local_model_path, local_files_only=True)
    print("✓ Tokenizer 加载成功")

    # 加载模型时直接指定设备
    model = BertForSequenceClassification.from_pretrained(
        local_model_path,
        local_files_only=True,
        num_labels=2
    ).to(device)  # 直接移动到设备

    print("✓ 模型加载成功并已移动到设备")

    optimizer = AdamW(model.parameters(), lr=2e-5)
    print("✓ 优化器初始化成功")

except Exception as e:
    print(f"加载失败: {e}")
    sys.exit(1)

# 读取数据
try:
    train = pd.read_csv(r"/root/autodl-tmp/Downloads/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(r"/root/autodl-tmp/Downloads/testData.tsv", header=0, delimiter="\t", quoting=3)
    print("✓ 数据加载成功")
except Exception as e:
    print(f"数据加载失败: {e}")
    sys.exit(1)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


# 自定义DataLoader来确保数据直接加载到GPU
class GPUDataloader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            yield {key: val.to(self.device) for key, val in batch.items()}

    def __len__(self):
        return len(self.dataloader)


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # 准备数据
    train_texts, train_labels, test_texts = [], [], []
    for i, review in enumerate(train["review"]):
        train_texts.append(review)
        train_labels.append(train['sentiment'][i])

    for review in test['review']:
        test_texts.append(review)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=.2, random_state=42
    )

    print("使用本地tokenizer进行编码...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    train_dataset = TrainDataset(train_encodings, train_labels)
    val_dataset = TrainDataset(val_encodings, val_labels)
    test_dataset = TestDataset(test_encodings)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 使用GPU优化的DataLoader
    train_loader_gpu = GPUDataloader(train_loader, device)
    val_loader_gpu = GPUDataloader(val_loader, device)
    test_loader_gpu = GPUDataloader(test_loader, device)

    optim = AdamW(model.parameters(), lr=5e-5)

    # 清空GPU缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # 训练循环
    for epoch in range(3):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0

        model.train()  # 训练模式
        with tqdm(total=len(train_loader_gpu), desc=f"Epoch {epoch}") as pbar:
            for batch in train_loader_gpu:
                n += 1
                optim.zero_grad()

                # 数据已经在GPU上（通过GPUDataloader）
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optim.step()

                preds = torch.argmax(outputs.logits, dim=1)
                acc = (preds == labels).float().mean()

                train_acc += acc.item()
                train_loss += loss.item()

                pbar.set_postfix({
                    'train_loss': f'{train_loss / n:.4f}',
                    'train_acc': f'{train_acc / n:.2f}',
                    'gpu_mem': f'{torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB' if device.type == 'cuda' else 'cpu'
                })
                pbar.update(1)

        # 验证循环
        model.eval()  # 评估模式
        with torch.no_grad():
            for batch in val_loader_gpu:
                m += 1
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = outputs.loss

                preds = torch.argmax(outputs.logits, dim=1)
                acc = (preds == labels).float().mean()

                val_acc += acc.item()
                val_losses += val_loss.item()

        end = time.time()
        runtime = end - start

        print(f'Epoch {epoch}: '
              f'train_loss: {train_loss / n:.4f}, '
              f'train_acc: {train_acc / n:.2f}, '
              f'val_loss: {val_losses / m:.4f}, '
              f'val_acc: {val_acc / m:.2f}, '
              f'time: {runtime:.2f}s')

        # 每个epoch后清空缓存（可选）
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 测试预测
    test_pred = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader_gpu), desc='Prediction') as pbar:
            for batch in test_loader_gpu:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                outputs = model(input_ids, attention_mask=attention_mask)
                test_pred.extend(torch.argmax(outputs.logits.cpu().data, dim=1).numpy().tolist())
                pbar.update(1)

    # 保存结果
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})

    # 确保结果目录存在
    os.makedirs("./result", exist_ok=True)
    result_output.to_csv("./result/bert_native.csv", index=False, quoting=3)

    # 最终清空GPU缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    logging.info('结果保存成功!')
    print("训练完成！")