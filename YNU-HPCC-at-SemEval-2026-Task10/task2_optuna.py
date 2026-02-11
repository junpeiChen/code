import json
import random
import numpy as np
import shutil
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
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
import optuna
import optuna.visualization as vis


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
# 8. Optuna调参函数
# ===============================
def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    num_epochs = trial.suggest_int('num_epochs', 3, 10)
    warmup_ratio = trial.suggest_uniform('warmup_ratio', 0.05, 0.2)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    activation_function = trial.suggest_categorical("activation_function", ["relu", "gelu"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

    model_name = r"D:\models\distilbert-base-uncased"

    # Prepare data
    train_file = r"D:\Pycharm_2025.2.4\test\train_rehydrated.jsonl"
    train_raw = load_and_filter_data(train_file)
    train_data, val_data = train_test_split(train_raw, test_size=0.1, random_state=42)
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    # Tokenize the datasets
    train_ds = tokenize_data(train_dataset, tokenizer)
    valid_ds = tokenize_data(val_dataset, tokenizer)

    # Encode the labels
    train_ds = encode_labels(train_ds, {"No": 0, "Yes": 1})
    valid_ds = encode_labels(valid_ds, {"No": 0, "Yes": 1})

    # Calculate class weights
    labels = train_ds["labels"]
    w0 = 1.0 / labels.count(0)
    w1 = 1.0 / labels.count(1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float)

    # Define model
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "No", 1: "Yes"},
        label2id={"No": 0, "Yes": 1},
    )

    # Adjust dropout rate
    model.config.hidden_dropout_prob = dropout_rate
    model.config.attention_probs_dropout_prob = dropout_rate

    # Add activation function (in a simple way)
    if activation_function == "relu":
        model.config.activation_function = "relu"
    else:
        model.config.activation_function = "gelu"

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"optuna_model/{trial.number}",  # 每次训练都使用不同的文件夹保存模型
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",  # 每个 epoch 保存模型
        load_best_model_at_end=True,
        logging_dir="./logs_optuna",
        report_to="none",
    )

    trainer = RDropTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.class_weights = class_weights

    # 训练并保存当前模型
    trainer.train()

    # 获取当前模型的准确率
    metrics = trainer.evaluate()
    acc = metrics["eval_accuracy"]
    print(f"Trial {trial.number} accuracy: {acc:.4f}")

    # 使用准确率来创建文件夹名称
    model_path = f"optuna_model/best_model_{acc:.4f}"  # 根据准确率命名文件夹
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # 保存当前 trial 的模型
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # 返回当前试验的准确率
    return acc


# ===============================
# 9. Optuna优化流程
# ===============================
if __name__ == "__main__":
    # 创建一个 Study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)  # 运行10次实验

    print(f"Best trial: {study.best_trial.params}")

    # 获取最佳模型并保存
    best_model_path = f"optuna_model/best_model_{study.best_trial.value:.4f}"
    best_model_dir = f"optuna_best_model"

    # 如果最佳模型存在，则复制到新目录
    if os.path.exists(best_model_path):
        if os.path.exists(best_model_dir):
            shutil.rmtree(best_model_dir)
        shutil.copytree(best_model_path, best_model_dir)
        print("Best model saved →", best_model_dir)
    else:
        print(f"Error: Best model path '{best_model_path}' does not exist.")

    # =========================
    # 可视化调参过程
    # =========================
    # Plot Optuna's hyperparameter optimization process
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig('optimization_history.png', dpi=600)
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig('param_importances.png', dpi=600)
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.savefig('parallel_coordinate.png', dpi=600)
