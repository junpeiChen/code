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


test = pd.read_csv(r"D:\360Downloads\testData.tsv", header=0, delimiter="\t", quoting=3)

num_epochs = 10
embed_size = 300
num_hiddens = 120
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.05
device = torch.device('cuda:0')
use_gpu = True


# Read data from files
train = pd.read_csv(r"D:\360Downloads\labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv(r"D:\360Downloads\testData.tsv", header=0, delimiter="\t", quoting=3)


def review_to_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, "lxml").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    # if remove_stopwords:
    #     stops = set(stopwords.words("english"))
    #     words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
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
            for token in sentence:
                token_freqs[token] += 1

        for sentence in test:
            for token in sentence:
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
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]


def length_to_mask(lengths, device=None):
    """
    将序列长度转换为mask
    Args:
        lengths: 序列长度张量
        device: 设备类型
    Returns:
        mask: 布尔掩码张量
    """
    if device is None:
        device = lengths.device

    max_length = torch.max(lengths)
    mask = torch.arange(max_length, device=device).expand(lengths.shape[0], max_length) < lengths.unsqueeze(1)
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        # x形状: [seq_len, batch_size, d_model]
        device = x.device  # 获取输入张量的设备
        seq_len = x.size(0)

        # 动态计算位置编码 - 确保在正确的设备上
        pe = torch.zeros(seq_len, self.d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [seq_len, 1, d_model]

        x = x + pe
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
                 dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1, activation: str = "relu"):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim, dropout)

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
        # inputs形状: [batch_size, seq_len]
        device = inputs.device  # 获取输入张量的设备
        inputs = inputs.transpose(0, 1)  # 转换为 [seq_len, batch_size]

        # 词嵌入
        hidden_states = self.embeddings(inputs)  # [seq_len, batch_size, embedding_dim]

        # 位置编码
        hidden_states = self.position_embedding(hidden_states)

        # 创建mask - 确保在正确的设备上
        attention_mask = length_to_mask(lengths, device=device) == False

        # Transformer编码
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)

        # 使用平均池化
        hidden_states = hidden_states.mean(dim=0)  # [batch_size, embedding_dim]

        # 输出层
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(examples):
    max_len = 512  # 设置最大序列长度

    if isinstance(examples[0], tuple) and len(examples[0]) == 2:
        # 训练/验证数据: (tokens, label)
        inputs = [torch.tensor(ex[0][:max_len]) for ex in examples]  # 截断
        lengths = torch.tensor([len(seq) for seq in inputs])
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    else:
        # 测试数据: 只有tokens
        inputs = [torch.tensor(ex[:max_len]) for ex in examples]  # 截断
        lengths = torch.tensor([len(seq) for seq in inputs])
        targets = torch.zeros(len(examples), dtype=torch.long)

    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

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
    test_reviews = [vocab.convert_tokens_to_ids(sentence)
                     for sentence in clean_test_reviews]

    train_reviews, val_reviews, train_labels, val_labels = train_test_split(train_reviews, train_labels,
                                                                            test_size=0.2, random_state=0)

    net = Transformer(vocab_size=len(vocab), embedding_dim=embed_size, hidden_dim=num_hiddens, num_class=labels)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = TransformerDataset(train_reviews)
    val_set = TransformerDataset(val_reviews)
    test_set = TransformerDataset(test_reviews)

    train_iter = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        with tqdm(total=len(train_iter), desc='Epoch %d' % epoch) as pbar:
            for feature, lengths, label in train_iter:
                n += 1
                net.zero_grad()

                # 将所有张量移动到GPU
                feature = feature.to(device)
                lengths = lengths.to(device)
                label = label.to(device)

                score = net(feature, lengths)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()

                train_acc += accuracy_score(torch.argmax(score.cpu().data, dim=1), label.cpu())
                train_loss += loss

                pbar.set_postfix({
                    'epoch': '%d' % (epoch),
                    'train loss': '%.4f' % (train_loss.data / n),
                    'train acc': '%.2f' % (train_acc / n)
                })
                pbar.update(1)

            with torch.no_grad():
                for val_feature, val_lengths, val_label in val_iter:
                    m += 1
                    # 将所有验证张量移动到GPU
                    val_feature = val_feature.to(device)
                    val_lengths = val_lengths.to(device)
                    val_label = val_label.to(device)

                    val_score = net(val_feature, val_lengths)
                    val_loss = loss_function(val_score, val_label)
                    val_acc += accuracy_score(torch.argmax(val_score.cpu().data, dim=1), val_label.cpu())
                    val_losses += val_loss

            end = time.time()
            runtime = end - start
            pbar.set_postfix({
                'epoch': '%d' % (epoch),
                'train loss': '%.4f' % (train_loss.data / n),
                'train acc': '%.2f' % (train_acc / n),
                'val loss': '%.4f' % (val_losses.data / m),
                'val acc': '%.2f' % (val_acc / m),
                'time': '%.2f' % (runtime)
            })
            # tqdm.write('{epoch: %d, train loss: %.4f, train acc: %.2f, val loss: %.4f, val acc: %.2f, time: %.2f}' %
            #       (epoch, train_loss.data / n, train_acc / n, val_losses.data / m, val_acc / m, runtime))

    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, test_lengths, _ in test_iter:
                # 将所有测试张量移动到GPU
                test_feature = test_feature.to(device)
                test_lengths = test_lengths.to(device)

                test_score = net(test_feature, test_lengths)
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())
                pbar.update(1)
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/transformer.csv", index=False, quoting=3)
    logging.info('result saved!')
