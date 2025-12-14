import json
import torch
import torch.nn as nn
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    TrainingArguments,
    Trainer,
)
from transformers import DataCollatorForTokenClassification
from datasets import Dataset


# ============================================================
# 1. 加载数据
# ============================================================
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    return data


# ============================================================
# 2. 简化标签
# ============================================================
def create_label_maps_simplified(marker_type):
    label_list = ["O", marker_type]
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    return label_to_id, id_to_label, len(label_list)


# ============================================================
# 3. Tokenize + 标签对齐（含 PAD → -100）
# ============================================================
def tokenize_and_align_labels_simplified(examples, tokenizer, label_to_id, marker_type):

    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
        return_offsets_mapping=True
    )

    labels = []
    all_markers = examples.get("markers", [])

    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        example_labels = [0] * len(offsets)  # 初始化为 'O' 标签
        example_markers = all_markers[i] if i < len(all_markers) else []

        # 对齐 markers
        for marker in example_markers:
            if marker["type"] == marker_type:
                start_char = marker["startIndex"]
                end_char = marker["endIndex"]
                marker_label = label_to_id[marker_type]

                for token_idx, (start, end) in enumerate(offsets):
                    if start is None or end is None:
                        continue
                    # 判断 token 是否落在 span 中
                    if start < end_char and end > start_char:
                        example_labels[token_idx] = marker_label

        # 忽略 PAD（标签为 -100）
        input_ids = tokenized_inputs["input_ids"][i]
        for j, token_id in enumerate(input_ids):
            if token_id == tokenizer.pad_token_id:
                example_labels[j] = -100  # 设置PAD对应的标签为-100

        labels.append(example_labels)

    tokenized_inputs["labels"] = labels
    tokenized_inputs.pop("offset_mapping")

    return tokenized_inputs


# ============================================================
# 4. WeightedLossTrainer（兼容新版 Trainer）
# ============================================================
class WeightedLossTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs["logits"]

        # class weight（提升正类 recall）
        weight = torch.tensor([1.0, 5.0]).to(logits.device)  # "O" 和 "marker_type" 的权重

        loss_fct = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=-100  # 忽略PAD的标签
        )

        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


# ============================================================
# 5. 主程序
# ============================================================
if __name__ == "__main__":

    train_file = r"D:\Pycharm_2025.2.4\test\train_rehydrated.jsonl"
    model_path = r"D:\models\distilbert-base-uncased"

    output_dir_base = "distilbert-single-type-simplified"

    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 4  # 避免过拟合

    marker_types_to_train = ["Action", "Actor", "Effect", "Evidence", "Victim"]

    # 加载数据
    data = load_data(train_file)
    dataset = Dataset.from_list(data)

    # train/val split
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

    # ================================
    # 逐个训练 marker 类型
    # ================================
    for marker_type in marker_types_to_train:

        print(f"\n======= Training model for: {marker_type} =======")

        # 创建标签
        label_to_id, id_to_label, num_labels = create_label_maps_simplified(marker_type)

        # tokenize
        tokenized_dataset = dataset_split.map(
            tokenize_and_align_labels_simplified,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "label_to_id": label_to_id, "marker_type": marker_type}
        )

        # 加载 DistilBERT
        model = DistilBertForTokenClassification.from_pretrained(
            model_path,
            num_labels=num_labels
        )

        # 输出目录
        output_dir = f"{output_dir_base}-{marker_type}"

        # TrainingArguments（允许保存模型）
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",     # 每个 epoch 保存一次（你最初代码是默认行为）
            logging_dir=f"./logs-{marker_type}",
            report_to="none"
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            processing_class=tokenizer,
        )

        # 开始训练
        trainer.train()

        # ========== 保存模型（与你最开始的代码保存格式一模一样） ==========
        trainer.save_model(output_dir)        # 保存模型（权重 + config）
        tokenizer.save_pretrained(output_dir) # 保存 tokenizer 文件

        print(f"Training + Saved model for {marker_type} at: {output_dir}")
