import json
import random
import numpy as np
import shutil
import torch
from torch import nn
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score
import os


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
# 2. 数据增强
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
# 3. 加载 + 清洗 + 过滤
# ===============================
def load_and_filter_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "conspiracy" in obj and obj["conspiracy"] in ["Yes", "No"]:
                    txt = augment_text(clean_text(obj["text"]))
                    data.append({"text": txt, "conspiracy": obj["conspiracy"]})
            except:
                pass
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
    return dataset.map(lambda e: {"labels": [label_to_id[i] for i in e["conspiracy"]]}, batched=True)


# ===============================
# 6. R-Drop Trainer
# ===============================
class RDropTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        ce = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device), label_smoothing=0.1)

        out1 = model(**inputs)
        out2 = model(**inputs)

        logits1 = out1.logits
        logits2 = out2.logits

        ce_loss = (ce(logits1, labels) + ce(logits2, labels)) / 2

        # KL Loss
        kl = nn.KLDivLoss(reduction="batchmean")
        p1 = torch.softmax(logits1, dim=-1)
        p2 = torch.softmax(logits2, dim=-1)
        log_p1 = torch.log_softmax(logits1, dim=-1)
        log_p2 = torch.log_softmax(logits2, dim=-1)

        kl_loss = (kl(log_p1, p2) + kl(log_p2, p1)) / 2

        loss = ce_loss + 0.5 * kl_loss
        return (loss, out1) if return_outputs else loss


# ===============================
# 7. 评估函数
# ===============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


# ===============================
# 主训练函数
# ===============================
def train_one_seed(seed, train_ds, valid_ds, tokenizer, model_name, class_weights):

    output_dir = f"model_seed_{seed}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"./logs_seed_{seed}",
        seed=seed,
        report_to="none",
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0:"No",1:"Yes"},
        label2id={"No":0,"Yes":1},
    )

    trainer = RDropTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.class_weights = class_weights

    trainer.train()

    # 保存模型为 Hugging Face 格式
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 读取验证集 accuracy
    metrics = trainer.evaluate()
    acc = metrics["eval_accuracy"]

    return acc, output_dir


# ===============================
# Ensemble 推理
# ===============================
def soft_ensemble(models, tokenizer, valid_ds):

    all_logits = []
    for m in models:
        model = DistilBertForSequenceClassification.from_pretrained(m).to("cuda")
        trainer = Trainer(model=model, tokenizer=tokenizer)
        out = trainer.predict(valid_ds)
        all_logits.append(out.predictions)

    avg_logits = np.mean(all_logits, axis=0)
    preds = np.argmax(avg_logits, axis=-1)

    acc = accuracy_score(valid_ds["labels"], preds)
    return acc


# ===============================
# 主程序
# ===============================
if __name__ == "__main__":

    train_file = r"D:\Pycharm_2025.2.4\test\train_rehydrated.jsonl"
    model_name = r"D:\models\distilbert-base-uncased"

    raw = load_and_filter_data(train_file)
    dataset = Dataset.from_list(raw)

    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset_split["train"]
    valid_ds = dataset_split["test"]

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    train_ds = tokenize_data(train_ds, tokenizer)
    valid_ds = tokenize_data(valid_ds, tokenizer)
    train_ds = encode_labels(train_ds, {"No": 0, "Yes": 1})
    valid_ds = encode_labels(valid_ds, {"No": 0, "Yes": 1})

    # class weights
    labels = train_ds["labels"]
    w0 = 1.0 / labels.count(0)
    w1 = 1.0 / labels.count(1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float)

    # =========================
    # 训练 5 次
    # =========================
    results = {}
    for seed in [0, 1, 2, 3, 4]:
        print(f"\n======== Training SEED {seed} ========")
        acc, path = train_one_seed(seed, train_ds, valid_ds, tokenizer, model_name, class_weights)
        results[path] = acc
        print(f"> Seed {seed} ACC = {acc:.4f}")

    # =========================
    # 找到 5 次中效果最好的模型
    # =========================
    best_model = max(results, key=results.get)
    print("\n============================")
    print(" Best model:", best_model)
    print("============================")

    # 保存到标准目录
    if os.path.exists("best_model"):
        shutil.rmtree("best_model")
    shutil.copytree(best_model, "best_model")

    print("Best model saved → best_model/")

    # =========================
    # Soft Ensemble
    # =========================
    print("\nRunning Soft Ensemble...")
    ensemble_acc = soft_ensemble(list(results.keys()), tokenizer, valid_ds)
    print(f"\nSoft Ensemble Accuracy = {ensemble_acc:.4f}")

