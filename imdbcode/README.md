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

| 模型 | 测试集最高准确率 | Epoch次数 | Kaggle分数 | 备注 |
|------|--------------|-----------|------------|------|
| attention_lstm | 0.82 | 10 | 0.81 | 注意力机制+LSTM |
| bert_native | 0.92 | 3 | 0.87 | 原生BERT实现 |
| bert_scratch | 0.93 | 3 | 0.93 | 从头训练的BERT |
| bert_trainer | 0.93 | 3 | 0.94 | 使用Trainer的BERT |
| capsule_lstm | 0.90 | 7 | 0.87 | 胶囊网络+LSTM |
| cnn | 0.87 | 10 | 0.86 | 卷积神经网络 |
| cnnlstm | 0.86 | 10 | 0.85 | CNN+LSTM混合模型 |
| distilbert_native | 0.91 | 3 | 0.92 | 原生DistilBERT |
| distilbert_trainer | 0.93 | 3 | 0.93 | 使用Trainer的DistilBERT |
| gru | 0.84 | 10 | 0.84 | 门控循环单元 |
| lstm | 0.89 | 10 | 0.88 | 长短期记忆网络 |
| roberta_trainer | 0.94 | 1 | 0.95 | 使用Trainer的RoBERTa |
| transformer | 0.80 | 10 | 0.79 | Transformer编码器 |
| bert-rdrop | 0.91 | 3 | 0.91 | - |
| bert-scl-trainer | 0.93 | 3 | 0.93 | - |
| modernbert-unsloth | 0.95 | 3 | 0.95 | - |
## 📊 结果分析
基于提供的实验结果，所有模型在IMDB电影评论情感分析任务上均表现出色，测试集准确率分布在80%-94%之间，体现了现代深度学习模型在文本分类任务上的强大能力。
1. 预训练模型的绝对优势
2. 合适的超参数调优对传统模型至关重要
3. 可能受益于更好的初始化或正则化策略
## 模型性能对比

### 1. DeBERTa-base 微调结果（监督学习）

| 训练样本数 | 验证集准确率 |
|-----------|-------------|
| 16        | 53.78%      |
| 64        | 66.06%      |
| 256       | 86.47%      |
| 1024      | 91.40%      |
| 67349（全量） | 93.69%      |

### 2. 大语言模型学习结果

| 模型 | 参数量 | 准确率 |
|------|--------|--------|
| TinyLlama-1.1B | 1.1B | 33.33% |
| Qwen-0.5B | 0.5B | 46.50% |
| Phi-2-2.7B | 2.7B | 64.95% |
| Mistral-7B | 7B | 87.31% |
| Gemma-2B | 2B | 91.50% |

*测试样本量：200*

### Qwen2.5-0.5B 电影评论情感分类项目
本项目使用 Qwen2.5-0.5B-Instruct 模型，通过先进的提示词工程和微调技术，在IMDb电影评论情感分类任务上实现了 89% 的验证准确率。
| 模型 | 参数量 | 准确率 |
|------|--------|--------|
| Qwen-0.5B | 0.5B | 89.43% |


# 用更大的模型合成SST的数据，引导小模型版本的Llama、Qwen、Gemma、Phi-4和Mistral进行微调

为了提高情感分析任务（SST-2）上的性能，我们首先使用更大的预训练模型生成合成数据。通过基于这些大模型生成的合成数据，我们可以为小模型（例如 Llama、Qwen、Gemma、Phi-4 和 Mistral）进行微调，以使其适应特定的指令学习任务。

## 步骤概览

1. **生成合成数据**：使用大规模预训练模型（例如 `google/gemma-7b`）生成包含情感分析标签（正面/负面）的合成文本数据。
2. **格式化数据**：将合成数据格式化为适合微调的小模型的格式，通常采用指令-响应（Instruction-Response）格式。
3. **微调小模型**：使用生成的合成数据对小模型（例如 `meta-llama/Llama-3B`、`Qwen2.5-1.5B`、`google/gemma-2-2b-it`、`microsoft/phi-2` 和 `Voxtral-Mini-3B-2507` 等）进行微调，以便它们能在 SST-2 情感分析任务上发挥更好的表现。

## 模型列表

| **Model**                   | **Parameters** | **Accuracy** |
|-----------------------------|----------------|--------------|
| Llama-3.2-3B                | 3B             | 87.23%       |
| Qwen2.5-1.5B                | 1.5B           | 83.56%       |
| Gemma-2-2B-it               | 2B             | 89.12%       |
| Phi-2                       | 3B             | 90.31%       |
| Voxtral-Mini-3B-2507        | 3B             | 86.78%       |

## 目标

通过微调较小的模型，我们希望能够优化它们在特定任务上的表现，特别是在情感分析任务中，以便能够在资源有限的情况下提供高效的推理能力和精度。
