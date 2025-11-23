import torch
import numpy as np
from transformers import DebertaTokenizer, DebertaForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from datasets import load_dataset, Dataset
import evaluate
from sklearn.metrics import accuracy_score
import random


# 设置随机种子确保结果可重现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# 加载SST-2数据集
print("Loading SST-2 dataset...")
dataset = load_dataset("glue", "sst2")

# 加载DeBERTa tokenizer和模型
print("Loading DeBERTa model and tokenizer...")
model_name = "microsoft/deberta-base"
tokenizer = DebertaTokenizer.from_pretrained(model_name)
model = DebertaForSequenceClassification.from_pretrained(model_name, num_labels=2)


# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(
        examples['sentence'],
        truncation=True,
        padding=True,
        max_length=256
    )


# 预处理数据集
print("Preprocessing dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./deberta-sst2-results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=500,
    report_to=None,  # 禁用wandb等记录器
)

# 加载准确率评估指标
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


# 实验1：使用全部训练集微调
print("\n" + "=" * 50)
print("实验1: 使用全部训练集微调DeBERTa-base")
print("=" * 50)

# 准备训练集和验证集
full_train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)

# 训练模型
print("开始训练...")
trainer.train()

# 在测试集上评估
print("在验证集上评估模型...")
eval_results = trainer.evaluate()
print(f"全部训练集微调后的验证集准确率: {eval_results['eval_accuracy']:.4f}")

# 实验2：使用不同样本数量的子集进行微调
sample_sizes = [16, 64, 256, 1024]

print("\n" + "=" * 50)
print("实验2: 使用不同样本数量的子集微调")
print("=" * 50)

results = {}

for sample_size in sample_sizes:
    print(f"\n--- 使用 {sample_size} 个样本进行微调 ---")

    # 从训练集中采样指定数量的样本
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(sample_size))

    # 重新初始化模型以确保公平比较
    model_small = DebertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 调整训练参数以适应小样本学习
    small_training_args = TrainingArguments(
        output_dir=f"./deberta-sst2-{sample_size}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,  # 小样本时需要更多epoch
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=f"./logs-{sample_size}",
        logging_steps=50,
        report_to=None,
    )

    # 创建Trainer
    trainer_small = Trainer(
        model=model_small,
        args=small_training_args,
        train_dataset=small_train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 训练
    print(f"使用 {sample_size} 个样本训练中...")
    trainer_small.train()

    # 评估
    eval_results_small = trainer_small.evaluate()
    accuracy = eval_results_small['eval_accuracy']
    results[sample_size] = accuracy
    print(f"{sample_size} 个样本微调后的验证集准确率: {accuracy:.4f}")

# 打印最终结果
print("\n" + "=" * 60)
print("最终实验结果汇总")
print("=" * 60)
print(f"全部训练集 ({len(tokenized_datasets['train'])} 样本): {eval_results['eval_accuracy']:.4f}")
for sample_size, accuracy in results.items():
    print(f"{sample_size:4d} 个样本: {accuracy:.4f}")

# 保存结果到文件
with open("deberta_sst2_results.txt", "w") as f:
    f.write("DeBERTa-base在SST-2数据集上的微调结果\n")
    f.write("=" * 50 + "\n")
    f.write(f"全部训练集 ({len(tokenized_datasets['train'])} 样本): {eval_results['eval_accuracy']:.4f}\n")
    for sample_size, accuracy in results.items():
        f.write(f"{sample_size:4d} 个样本: {accuracy:.4f}\n")

print("\n结果已保存到 deberta_sst2_results.txt")