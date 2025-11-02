import os
import sys
import logging
import time

import pandas as pd
import torch.optim as optim

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 修改为DistilBERT的本地路径
LOCAL_MODEL_PATH = r"/root/autodl-tmp/distilbert-base-uncased"  # 注意路径变化

train = pd.read_csv(r"/root/autodl-tmp/Downloads/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv(r"/root/autodl-tmp/Downloads/testData.tsv", header=0, delimiter="\t", quoting=3)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        if labels:
            self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, num_samples=0):
        self.encodings = encodings
        self.num_samples = num_samples

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train_texts, train_labels, test_texts = [], [], []
    for i, review in enumerate(train["review"]):
        train_texts.append(review)
        train_labels.append(train['sentiment'][i])

    for review in test['review']:
        test_texts.append(review)

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    # 使用DistilBERT tokenizer（现在类型匹配了）
    tokenizer = DistilBertTokenizerFast.from_pretrained(LOCAL_MODEL_PATH)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = TrainDataset(train_encodings, train_labels)
    val_dataset = TrainDataset(val_encodings, val_labels)
    test_dataset = TestDataset(test_encodings, num_samples=len(test_texts))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 使用DistilBERT模型（现在类型匹配了）
    model = DistilBertForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

    optim = optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}") as pbar:
            for batch in train_loader:
                n += 1
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optim.step()
                train_acc += accuracy_score(torch.argmax(outputs.logits.cpu().data, dim=1), labels.cpu())
                train_loss += loss.cpu()

                if n % 100 == 0:
                    pbar.set_postfix({
                        'loss': '%.4f' % (train_loss.data / n),
                        'acc': '%.4f' % (train_acc / n)
                    })
                pbar.update(1)

            with torch.no_grad():
                for batch in val_loader:
                    m += 1
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss = outputs.loss
                    val_acc += accuracy_score(torch.argmax(outputs.logits.cpu().data, dim=1), labels.cpu())
                    val_losses += val_loss.cpu()

            end = time.time()
            runtime = end - start

            epoch_train_loss = train_loss.data / n
            epoch_train_acc = train_acc / n
            epoch_val_loss = val_losses.data / m
            epoch_val_acc = val_acc / m

            print(f'\n=== Epoch {epoch} 完成 ===')
            print(f'训练损失: {epoch_train_loss:.4f} | 训练准确率: {epoch_train_acc:.4f}')
            print(f'验证损失: {epoch_val_loss:.4f} | 验证准确率: {epoch_val_acc:.4f}')
            print(f'耗时: {runtime:.2f}秒\n')

    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Prediction') as pbar:
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                test_pred.extend(torch.argmax(outputs.logits.cpu().data, dim=1).numpy().tolist())
                pbar.update(1)

    os.makedirs("./result", exist_ok=True)
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/distilbert_native.csv", index=False, quoting=3)
    logging.info('result saved!')