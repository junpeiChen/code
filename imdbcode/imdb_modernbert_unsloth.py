import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 在导入 unsloth 之前先导入 torch 并检查版本
import torch

print(f"PyTorch version: {torch.__version__}")

# 尝试修复版本解析问题
try:
    import unsloth
except AttributeError as e:
    if "'NoneType' object has no attribute 'group'" in str(e):
        # 手动设置 torch 版本信息
        import re
        from importlib.metadata import version as importlib_version

        torch_version_str = importlib_version("torch")
        match = re.match(r"[0-9\.]{3,}", torch_version_str)
        if match:
            torch_version = match.group(0).split(".")
        else:
            # 如果正则匹配失败，使用默认版本
            torch_version = ["2", "1", "0"]

        # 重新导入 unsloth
        import unsloth
    else:
        raise

import sys
import logging

import pandas as pd
import numpy as np

from unsloth import FastModel, FastLanguageModel
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from datasets import Dataset

from sklearn.model_selection import train_test_split

train = pd.read_csv(r"D:\360Downloads\labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv(r"D:\360Downloads\testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train, val = train_test_split(train, test_size=.2)

    # train = train[0:20]

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)

    # model_name = 'answerdotai/ModernBERT-large'
    model_name = r"D:\models\deberta-v3-small"
    NUM_CLASSES = 2

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,
        max_seq_length=2048,
        dtype=None,
        auto_model=AutoModelForSequenceClassification,
        num_labels=NUM_CLASSES,
        gpu_memory_utilization=0.8  # Reduce if out of memory
    )

    model = FastModel.get_peft_model(
        model,
        r=16,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=32,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
        target_modules="all-linear",  # Optional now! Can specify a list if needed
        task_type="SEQ_CLS",
    )

    print("model parameters:" + str(sum(p.numel() for p in model.parameters())))

    # make all parameters trainable
    # for param in model.parameters():
    #     param.requires_grad = True


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}


    def tokenize_function(examples):
        # return tokenizer(examples['text'])
        return tokenizer(examples['text'], max_length=512, truncation=True, padding=True)  # 添加了padding


    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    print(test_dataset)

    training_args = TrainingArguments(
        output_dir="./results",  # 添加输出目录
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_torch",  # 直接使用字符串
        learning_rate=2e-5,
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        num_train_epochs=3,  # bert-style models usually need more than 1 epoch
        save_strategy="epoch",
        # report_to="wandb",
        # report_to="none",
        # group_by_length=True,
        # eval_strategy="no",
        eval_strategy="epoch",  # 修正参数名
        # eval_steps=0.25,
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,  # 添加这个参数
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,  # 修正参数名
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer_stats = trainer.train()

    print(trainer_stats)

    model = model.eval()
    FastLanguageModel.for_inference(model)

    prediction_outputs = trainer.predict(test_dataset)
    print(prediction_outputs)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    # 确保输出目录存在
    os.makedirs("./result", exist_ok=True)
    result_output.to_csv("./result/deberta_unsloth.csv", index=False, quoting=3)
    logging.info('result saved!')