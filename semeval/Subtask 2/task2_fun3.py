import json
import random
import numpy as np
import torch
from torch import nn
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset


# ===============================
# 1. 数据清洗
# ===============================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.replace("\n", " ").replace("\t", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


# ===============================
# 2. 数据增强（简单可靠）
# ===============================
def augment_text(text):
    words = text.split()
    if len(words) <= 4:
        return text
    if random.random() < 0.2:
        idx = random.randint(0, len(words) - 1)
        del words[idx]
        return " ".join(words)
    return text


# ===============================
# 3. 加载 + 清洗 + 过滤 + 增强
# ===============================
def load_and_filter_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if "conspiracy" in item and item["conspiracy"] in ["Yes", "No"]:
                    text = clean_text(item["text"])
                    text = augment_text(text)
                    data.append({"text": text, "conspiracy": item["conspiracy"]})
            except:
                continue
    return data


# ===============================
# 4. Tokenize
# ===============================
def tokenize_data(dataset, tokenizer):
    return dataset.map(lambda e: tokenizer(e["text"], truncation=True), batched=True)


# ===============================
# 5. Encode labels
# ===============================
def encode_labels(dataset, label_to_id):
    return dataset.map(
        lambda e: {"labels": [label_to_id[label] for label in e["conspiracy"]]},
        batched=True
    )


# ===============================
# 6. R-Drop + Label Smoothing Trainer
# ===============================
class RDropTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]

        # --- 关键：Label Smoothing ---
        ce_loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device), label_smoothing=0.1)

        # --- 两次前向（R-Drop 关键）---
        outputs1 = model(**inputs)
        outputs2 = model(**inputs)

        logits1 = outputs1.logits
        logits2 = outputs2.logits

        ce_loss = (ce_loss_fct(logits1, labels) + ce_loss_fct(logits2, labels)) / 2

        # --- KL 损失（R-Drop 关键）---
        kl_loss_fct = nn.KLDivLoss(reduction="batchmean")

        log_prob1 = torch.nn.functional.log_softmax(logits1, dim=-1)
        prob2 = torch.nn.functional.softmax(logits2, dim=-1)

        log_prob2 = torch.nn.functional.log_softmax(logits2, dim=-1)
        prob1 = torch.nn.functional.softmax(logits1, dim=-1)

        kl_loss = (kl_loss_fct(log_prob1, prob2) + kl_loss_fct(log_prob2, prob1)) / 2

        loss = ce_loss + 0.5 * kl_loss    # R-Drop loss

        if return_outputs:
            return loss, outputs1
        return loss


# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    train_file = r"D:\Pycharm_2025.2.4\test\train_rehydrated.jsonl"
    model_name = r"D:\models\distilbert-base-uncased"
    output_dir = "distilbert-conspiracy-rdrop"

    label_to_id = {"No": 0, "Yes": 1}
    id_to_label = {0: "No", 1: "Yes"}

    # 加载数据
    raw_data = load_and_filter_data(train_file)
    dataset = Dataset.from_list(raw_data)

    # 划分 train/valid
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset_split["train"]
    valid_ds = dataset_split["test"]

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    train_ds = tokenize_data(train_ds, tokenizer)
    valid_ds = tokenize_data(valid_ds, tokenizer)
    train_ds = encode_labels(train_ds, label_to_id)
    valid_ds = encode_labels(valid_ds, label_to_id)

    # 类别权重
    labels = train_ds["labels"]
    num_no = labels.count(0)
    num_yes = labels.count(1)

    weight_no = 1.0 / num_no
    weight_yes = 1.0 / num_yes
    class_weights = torch.tensor([weight_no, weight_yes], dtype=torch.float)

    # 模型加载 + 增加 dropout
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id_to_label,
        label2id=label_to_id
    )
    model.config.dropout = 0.2
    model.config.attention_dropout = 0.2

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=15,
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        logging_dir="./logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    # Trainer（R-Drop + Label Smoothing + Weighted Loss）
    trainer = RDropTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    trainer.class_weights = class_weights  # 权重注入

    print("Training model...")
    trainer.train()
    print("Training finished.")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved:", output_dir)
