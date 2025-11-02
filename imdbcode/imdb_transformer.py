import logging
import os
import sys
import time
import math
import re

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        # 预先计算位置编码，但会在forward中动态处理
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        seq_len = x.size(0)
        if seq_len > self.max_len:
            # 如果序列长度超过max_len，使用动态计算
            pe = self._dynamic_positional_encoding(seq_len, x.size(2), x.device)
            x = x + pe
        else:
            x = x + self.pe[:seq_len]
        return self.dropout(x)

    def _dynamic_positional_encoding(self, seq_len, d_model, device):
        """动态计算位置编码"""
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1)


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
                 dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1, max_len=5000, activation: str = "relu"):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)

        # 编码层：使用Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出层
        self.output = nn.Linear(embedding_dim, num_class)

    def forward(self, inputs, lengths):
        # inputs shape: (batch_size, seq_len)
        # 转置为 (seq_len, batch_size)
        inputs = inputs.transpose(0, 1)

        # 词嵌入 + 位置编码
        hidden_states = self.embeddings(inputs)  # (seq_len, batch_size, embedding_dim)
        hidden_states = self.position_embedding(hidden_states)

        # 创建注意力mask
        attention_mask = self.create_mask(lengths, inputs.size(0))

        # Transformer编码
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)

        # 使用平均池化
        hidden_states = hidden_states.mean(dim=0)  # (batch_size, embedding_dim)

        # 输出层
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

    def create_mask(self, lengths, max_len):
        """创建padding mask"""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len).expand(batch_size, max_len).to(lengths.device) >= lengths.unsqueeze(1)
        return mask


def length_to_mask(lengths):
    max_length = torch.max(lengths)
    mask = torch.arange(max_length).expand(lengths.shape[0], max_length) < lengths.unsqueeze(1)
    return mask


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets


def review_to_wordlist(review, remove_stopwords=False):
    review_text = re.sub(r'<[^>]+>', ' ', review)
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    return ' '.join(words)


class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, train, test, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in train:
            for token in sentence.split():
                token_freqs[token] += 1

        for sentence in test:
            for token in sentence.split():
                token_freqs[token] += 1

        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() \
                        if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            tokens = tokens.split()
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]


def save_model(model, optimizer, epoch, loss, path):
    """保存模型和训练状态"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    logging.info(f"Model saved to {path}")


def load_model(model, optimizer, path):
    """加载模型和训练状态"""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logging.info(f"Model loaded from {path}, epoch: {epoch}, loss: {loss:.4f}")
        return epoch, loss
    else:
        logging.info(f"No checkpoint found at {path}")
        return 0, float('inf')


if __name__ == '__main__':
    # 参数设置
    num_epochs = 10
    embed_size = 300
    num_hiddens = 120
    num_layers = 2
    bidirectional = True
    batch_size = 32
    labels = 2
    lr = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 创建结果目录
    os.makedirs("./result", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # 读取数据
    train = pd.read_csv("/root/autodl-tmp/Downloads/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("/root/autodl-tmp/Downloads/testData.tsv", header=0, delimiter="\t", quoting=3)

    clean_train_reviews, train_labels = [], []
    for i, review in enumerate(train["review"]):
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=False))
        train_labels.append(train["sentiment"][i])

    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=False))

    vocab = Vocab.build(clean_train_reviews, clean_test_reviews)

    train_reviews = [(vocab.convert_tokens_to_ids(sentence), train_labels[i])
                     for i, sentence in enumerate(clean_train_reviews)]

    # 限制序列最大长度以避免内存问题
    max_sequence_length = 512
    train_reviews_trimmed = []
    for tokens, label in train_reviews:
        if len(tokens) > max_sequence_length:
            tokens = tokens[:max_sequence_length]
        train_reviews_trimmed.append((tokens, label))

    train_reviews, val_reviews = train_test_split(train_reviews_trimmed, test_size=0.2, random_state=0)

    # 准备测试数据
    test_reviews = [vocab.convert_tokens_to_ids(sentence) for sentence in clean_test_reviews]
    test_reviews_trimmed = []
    for tokens in test_reviews:
        if len(tokens) > max_sequence_length:
            tokens = tokens[:max_sequence_length]
        test_reviews_trimmed.append(tokens)

    # 创建模型
    net = Transformer(
        vocab_size=len(vocab),
        embedding_dim=embed_size,
        hidden_dim=num_hiddens,
        num_class=labels,
        max_len=5000
    )
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # 创建数据集
    train_set = TransformerDataset(train_reviews)
    val_set = TransformerDataset(val_reviews)
    test_set = TransformerDataset([(tokens, 0) for tokens in test_reviews_trimmed])  # 为测试集添加伪标签

    train_iter = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

    # 训练模型
    best_val_loss = float('inf')
    best_model_path = "./models/best_transformer_model.pth"

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0

        # 训练阶段
        net.train()
        with tqdm(total=len(train_iter), desc='Epoch %d' % epoch) as pbar:
            for feature, lengths, label in train_iter:
                n += 1
                optimizer.zero_grad()
                feature = feature.to(device)
                lengths = lengths.to(device)
                label = label.to(device)

                score = net(feature, lengths)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()

                train_acc += accuracy_score(torch.argmax(score.cpu().data, dim=1), label.cpu())
                train_loss += loss.item()

                pbar.set_postfix({
                    'train loss': '%.4f' % (train_loss / n),
                    'train acc': '%.2f' % (train_acc / n)
                })
                pbar.update(1)

        # 验证阶段
        net.eval()
        with torch.no_grad():
            for val_feature, val_length, val_label in val_iter:
                m += 1
                val_feature = val_feature.to(device)
                val_length = val_length.to(device)
                val_label = val_label.to(device)
                val_score = net(val_feature, val_length)
                val_loss = loss_function(val_score, val_label)
                val_acc += accuracy_score(torch.argmax(val_score.cpu().data, dim=1), val_label.cpu())
                val_losses += val_loss.item()

        end = time.time()
        runtime = end - start

        avg_val_loss = val_losses / m
        print(f'Epoch {epoch}: train_loss: {train_loss / n:.4f}, train_acc: {train_acc / n:.2f}, '
              f'val_loss: {avg_val_loss:.4f}, val_acc: {val_acc / m:.2f}, time: {runtime:.2f}s')

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(net, optimizer, epoch, avg_val_loss, best_model_path)
            logging.info(f"New best model saved with val_loss: {avg_val_loss:.4f}")

    # 加载最佳模型进行预测
    logging.info("Loading best model for prediction...")
    load_model(net, optimizer, best_model_path)

    # 在测试集上进行预测
    net.eval()
    test_pred = []

    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Predicting on test set') as pbar:
            for test_feature, test_lengths, _ in test_iter:  # 忽略伪标签
                test_feature = test_feature.to(device)
                test_lengths = test_lengths.to(device)
                test_score = net(test_feature, test_lengths)
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())
                pbar.update(1)

    # 保存预测结果
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/transformer_predictions.csv", index=False, quoting=3)
    logging.info('Predictions saved to ./result/transformer_predictions.csv')

    # 保存训练和验证的详细结果
    training_stats = {
        'best_val_loss': best_val_loss,
        'final_train_acc': train_acc / n,
        'final_val_acc': val_acc / m,
        'vocab_size': len(vocab),
        'embed_size': embed_size,
        'num_layers': num_layers,
        'num_epochs': num_epochs
    }

    # 保存训练统计信息
    stats_df = pd.DataFrame([training_stats])
    stats_df.to_csv("./result/training_statistics.csv", index=False)
    logging.info('Training statistics saved to ./result/training_statistics.csv')

    # 打印最终结果摘要
    logging.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Final training accuracy: {train_acc / n:.4f}")
    logging.info(f"Final validation accuracy: {val_acc / m:.4f}")
    logging.info(f"Test predictions: {len(test_pred)} samples")
    logging.info(f"Positive sentiment ratio in test set: {sum(test_pred) / len(test_pred):.4f}")