# IMDB 电影评论情感分析

本项目使用多种机器学习模型对 IMDB 电影评论进行情感分析，预测评论的情感倾向（正面/负面）。
## 📊 项目概述

- **任务类型**：二分类情感分析
- **数据来源**：IMDB 电影评论数据集
- **目标**：根据电影评论内容预测情感倾向（0=负面，1=正面）
- ## 📁 数据集

### 训练数据 (`labeledTrainData.tsv`)
- 25,000 条带标签的影评
- 包含以下列：
  - `id`: 评论唯一标识
  - `sentiment`: 情感标签 (0=负面, 1=正面)
  - `review`: 评论文本

### 测试数据 (`testData.tsv`)
- 25,000 条未标记的影评
- 包含以下列：
  - `id`: 评论唯一标识
  - `review`: 评论文本

## 🤖 使用的模型

本项目实现了多种模型进行对比分析：
1. **attention_lstm**
2. **bert_native**
3. **bert_scratch**
4. **bert_trainer**
5. **capsule_lstm**
6. **cnn**
7. **cnnlstm**
8. **distilbert_native**
9. **distilbert_trainer**
10. **gru**
11. **lstm**
12. **roberta_trainer**
13. **transformer**

### 环境要求

```bash
pip install transformers
pip install datasets
pip install pandas
pip install numpy
pip install scikit-learn
pip install evaluate
pip intsall torch
```
## 📈 模型性能对比
### 准确率对比表

| 模型 | 测试集准确率 | Epoch次数 | Kaggle分数 | 备注 |
|------|--------------|-----------|------------|------|
| attention_lstm | 0.82 | 10 | 0.81 | 注意力机制+LSTM |
| bert_native | 0.92 | 3 | 0.87 | 原生BERT实现 |
| bert_scratch | 0.93 | 3 | 0.93 | 从头训练的BERT |
| bert_trainer | 0.93 | 3 | 0.94 | 使用Trainer的BERT |
| capsule_lstm | 0.50 | 10 | 0.50 | 胶囊网络+LSTM |
| cnn | 0.87 | 10 | 0.86 | 卷积神经网络 |
| cnnlstm | 0.86 | 10 | 0.85 | CNN+LSTM混合模型 |
| distilbert_native | 0.91 | 3 | 0.92 | 原生DistilBERT |
| distilbert_trainer | 0.93 | 3 | 0.93 | 使用Trainer的DistilBERT |
| gru | 0.84 | 10 | 0.84 | 门控循环单元 |
| lstm | 0.89 | 10 | 0.88 | 长短期记忆网络 |
| roberta_trainer | 0.94 | 1 | 0.95 | 使用Trainer的RoBERTa |
| transformer | 0.51 | 10 | 0.50 | Transformer编码器 |
## 📊 结果分析
