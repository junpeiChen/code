import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 数据路径 - 请确保路径正确
test = pd.read_csv("/root/autodl-tmp/Downloads/testData.tsv", header=0, delimiter="\t", quoting=3)

# 超参数调整
num_epochs = 10  # 增加训练轮数
embed_size = 300
num_hiddens = 256  # 增加隐藏层维度
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.001  # 降低学习率
weight_decay = 1e-4  # 添加权重衰减
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()

print(f"Using device: {device}")


class Capsule(nn.Module):
    def __init__(self, num_hiddens, bidirectional, num_capsule=10, dim_capsule=16, routings=3, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.bidirectional = bidirectional

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

        if self.bidirectional:
            self.W = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(1, num_hiddens * 2, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(1, num_hiddens, self.num_capsule * self.dim_capsule)))

        # 添加dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs):
        # inputs形状: (seq_len, batch_size, num_hiddens * 2)
        seq_len, batch_size, hidden_dim = inputs.shape

        # 重塑输入以便进行矩阵乘法
        inputs_reshaped = inputs.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
        inputs_reshaped = self.dropout(inputs_reshaped)

        u_hat_vecs = torch.matmul(inputs_reshaped, self.W)  # (batch_size, seq_len, num_capsule * dim_capsule)

        # 修正view操作的维度
        u_hat_vecs = u_hat_vecs.view((batch_size, seq_len, self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3).contiguous()  # (batch_size, num_capsule, seq_len, dim_capsule)

        # 使用更稳定的路由算法
        b = torch.zeros(batch_size, self.num_capsule, seq_len, device=inputs.device)

        for i in range(self.routings):
            c = F.softmax(b, dim=1)  # (batch_size, num_capsule, seq_len)
            outputs = self.squash(torch.sum(c.unsqueeze(-1) * u_hat_vecs, dim=2))  # bij,bijk->bik

            if i < self.routings - 1:
                b = b + torch.sum(outputs.unsqueeze(2) * u_hat_vecs, dim=-1)  # bik,bijk->bij

        return outputs  # (batch_size, num_capsule, dim_capsule)

    @staticmethod
    def squash(x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = s_squared_norm / (1 + s_squared_norm) / (torch.sqrt(s_squared_norm) + 1e-8)
        return scale * x


class Attention(nn.Module):
    """添加注意力机制"""

    def __init__(self, hidden_dim, bidirectional=True):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(self.hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: (seq_len, batch_size, hidden_dim)
        lstm_output = lstm_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)

        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)

        return context_vector


class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers, bidirectional, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional

        # 嵌入层
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = True  # 改为True进行微调

        # LSTM编码器
        self.encoder = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.num_hiddens,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=0.3 if num_layers > 1 else 0,  # 添加dropout
            batch_first=False
        )

        # 注意力机制
        self.attention = Attention(num_hiddens, bidirectional)

        # Capsule层
        self.capsule = Capsule(num_hiddens=self.num_hiddens, bidirectional=self.bidirectional)

        # 分类器
        capsule_output_dim = self.capsule.num_capsule * self.capsule.dim_capsule
        attention_output_dim = self.attention.hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(capsule_output_dim + attention_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, labels)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'embedding' not in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, inputs):
        # 嵌入层
        embeddings = self.embedding(inputs)  # (batch_size, seq_len) -> (batch_size, seq_len, embed_size)

        # LSTM编码
        lstm_out, (hidden, cell) = self.encoder(embeddings.permute(1, 0, 2))  # (seq_len, batch_size, hidden_dim*2)

        # 使用注意力机制
        attention_out = self.attention(lstm_out)  # (batch_size, hidden_dim*2)

        # Capsule网络
        capsule_out = self.capsule(lstm_out)  # (batch_size, num_capsule, dim_capsule)
        capsule_flat = capsule_out.view(capsule_out.size(0), -1)  # (batch_size, num_capsule * dim_capsule)

        # 结合注意力输出和Capsule输出
        combined = torch.cat([attention_out, capsule_flat], dim=1)

        # 分类
        outputs = self.classifier(combined)
        return outputs


def evaluate_model(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_acc += (preds == labels).sum().item()
            total_samples += features.size(0)

    return total_loss / total_samples, total_acc / total_samples


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'imdb_glove.pickle3')

    try:
        [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,
         vocab] = pickle.load(open(pickle_file, 'rb'))
        logging.info('data loaded!')

        # 打印数据形状信息
        logging.info(f'Train features shape: {train_features.shape}')
        logging.info(f'Train labels shape: {train_labels.shape}')
        logging.info(f'Val features shape: {val_features.shape}')
        logging.info(f'Test features shape: {test_features.shape}')
        logging.info(f'Weight shape: {weight.shape}')

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

    # 创建模型
    net = SentimentNet(
        embed_size=embed_size,
        num_hiddens=num_hiddens,
        num_layers=num_layers,
        bidirectional=bidirectional,
        weight=torch.FloatTensor(weight),
        labels=labels,
        use_gpu=use_gpu
    )
    net.to(device)

    # 打印模型结构
    logging.info("Model structure:")
    logging.info(net)

    # 计算参数数量
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # 损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)  # 使用AdamW
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)  # 学习率调度

    # 数据加载器
    train_set = TensorDataset(train_features, train_labels)
    val_set = TensorDataset(val_features, val_labels)
    test_set = TensorDataset(test_features)

    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 训练循环
    best_val_acc = 0.0
    patience_counter = 0
    patience = 5  # 早停耐心值

    for epoch in range(num_epochs):
        start = time.time()

        # 训练阶段
        net.train()
        train_loss, train_acc = 0.0, 0.0
        train_batches = 0

        with tqdm(total=len(train_iter), desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for feature, label in train_iter:
                feature, label = feature.to(device), label.to(device)

                optimizer.zero_grad()
                score = net(feature)
                loss = loss_function(score, label)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(score, dim=1)
                train_acc += (preds == label).float().mean().item()
                train_batches += 1

                pbar.set_postfix({
                    'train_loss': f'{train_loss / train_batches:.4f}',
                    'train_acc': f'{train_acc / train_batches:.4f}'
                })
                pbar.update(1)

        # 计算平均训练损失和准确率
        avg_train_loss = train_loss / train_batches
        avg_train_acc = train_acc / train_batches

        # 验证阶段
        val_loss, val_acc = evaluate_model(net, val_iter, loss_function, device)

        # 学习率调度
        scheduler.step(val_loss)

        end = time.time()
        runtime = end - start

        logging.info(f'Epoch {epoch + 1}: '
                     f'Train Loss: {avg_train_loss:.4f}, '
                     f'Train Acc: {avg_train_acc:.4f}, '
                     f'Val Loss: {val_loss:.4f}, '
                     f'Val Acc: {val_acc:.4f}, '
                     f'Time: {runtime:.2f}s')

        # 早停和模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(net.state_dict(), 'best_model.pth')
            logging.info(f'New best model saved with val_acc: {val_acc:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping at epoch {epoch + 1}')
                break

    # 加载最佳模型进行测试
    if os.path.exists('best_model.pth'):
        net.load_state_dict(torch.load('best_model.pth'))
        logging.info('Loaded best model for testing')

    # 预测
    test_pred = []
    net.eval()
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, in test_iter:
                test_feature = test_feature.to(device)
                test_score = net(test_feature)
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())
                pbar.update(1)

    # 保存结果
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    os.makedirs("./result", exist_ok=True)
    result_output.to_csv("./result/capsule_lstm_improved.csv", index=False, quoting=3)
    logging.info('Result saved!')
    logging.info(f'Best validation accuracy: {best_val_acc:.4f}')