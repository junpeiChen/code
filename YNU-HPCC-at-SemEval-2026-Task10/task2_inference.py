import json
import sys
import numpy as np
import os
import glob
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments,
)
from scipy.special import softmax

# 配置
MODEL_PATH = "best_model"
TEST_FILE = r"D:\Pycharm_2025.2.4\test\semeval\test_rehydrated.jsonl"
SUBMISSION_FILE = r"D:\Pycharm_2025.2.4\test\submission.jsonl"
MODEL_NAME = r"D:\models\distilbert-base-uncased"
LABEL_MAP = {0: "No", 1: "Yes"}
BATCH_SIZE = 64


def find_all_checkpoints(base_path):
    """
    查找所有保存的 checkpoint 文件夹
    """
    checkpoint_dirs = glob.glob(os.path.join(base_path, "checkpoint-*"))
    checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))
    return checkpoint_dirs


def load_competition_test_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                sample_id = item.get("_id", f"sample_{i}")
                data.append({
                    "unique_sample_id": sample_id,
                    "text": item.get("text", "")
                })
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line at index {i} in {file_path}: {line.strip()}")
    print(f"Loaded {len(data)} samples for inference.")
    return data


def tokenize_data(dataset, tokenizer):
    return dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, padding=True), batched=True)


def soft_voting(models, dataset, tokenizer):
    """
    使用软投票结合多个模型的预测结果
    """
    all_logits = []

    for model in models:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="./tmp_inference",
                per_device_eval_batch_size=BATCH_SIZE,
                report_to="none"
            ),
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        )

        # 获取预测结果
        predictions = trainer.predict(dataset)
        logits = predictions.predictions
        all_logits.append(softmax(logits, axis=-1))  # 计算 softmax 使输出为概率

    # 计算所有模型的平均预测概率
    avg_logits = np.mean(all_logits, axis=0)
    return np.argmax(avg_logits, axis=-1)  # 返回最大概率的类别


if __name__ == '__main__':
    # 1. 加载数据
    raw_data = load_competition_test_data(TEST_FILE)
    if not raw_data:
        print("Error: No data loaded. Cannot perform inference.")
        sys.exit(-1)

    # 转换为 Hugging Face Dataset 格式
    test_dataset = Dataset.from_list(raw_data)

    # 存储唯一ID，稍后用于生成提交文件
    unique_ids = test_dataset["unique_sample_id"]

    # 2. 查找所有检查点并加载模型
    checkpoint_dirs = find_all_checkpoints(MODEL_PATH)
    models = []

    print(f"Found {len(checkpoint_dirs)} checkpoints. Loading models...")

    for checkpoint_dir in checkpoint_dirs:
        try:
            model = DistilBertForSequenceClassification.from_pretrained(checkpoint_dir)
            models.append(model)
        except Exception as e:
            print(f"Error loading model from checkpoint {checkpoint_dir}: {e}")
            continue

    if not models:
        print("No models loaded. Exiting.")
        sys.exit(-1)

    # 3. 加载 tokenizer
    print(f"Loading tokenizer from {MODEL_NAME}...")
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(-1)

    # 4. 对数据进行 Tokenize
    tokenized_test_dataset = tokenize_data(test_dataset, tokenizer)

    # 删除模型不需要的列
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["unique_sample_id", "text"])

    # 5. 使用软投票进行推理
    print("Starting soft voting prediction...")
    predicted_class_ids = soft_voting(models, tokenized_test_dataset, tokenizer)

    # 6. 将预测结果映射为标签
    predicted_labels = [LABEL_MAP[int(id)] for id in predicted_class_ids]

    # 7. 保存结果为 JSONL 格式
    print(f"Saving {len(predicted_labels)} predictions to {SUBMISSION_FILE} (JSONL format)...")

    jsonl_lines = []
    for i, label in enumerate(predicted_labels):
        jsonl_obj = {
            "_id": unique_ids[i],
            "conspiracy": label
        }
        jsonl_lines.append(json.dumps(jsonl_obj))

    with open(SUBMISSION_FILE, 'w') as f:
        f.write('\n'.join(jsonl_lines) + '\n')

    print(f"Submission file '{SUBMISSION_FILE}' generated successfully.")
