import os
import sys
import logging
import datasets

import pandas as pd
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from peft import PrefixTuningConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 读取数据
train = pd.read_csv(r"/root/autodl-tmp/Downloads/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv(r"/root/autodl-tmp/Downloads/testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # 分割训练集和验证集
    train, val = train_test_split(train, test_size=.2)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    batch_size = 32

    model_id = r"/root/autodl-tmp/models/AI-ModelScope/gpt2"

    # 修复：使用 AutoTokenizer 而不是 DebertaV2Tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)


    # 如果 tokenizer 没有 pad_token，设置为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,  # 添加padding
            max_length=512  # 添加最大长度限制
        )


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,  # 二分类任务
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1}
    )


    # 配置 Prefix Tuning
    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=20,
        encoder_hidden_size=768  # 根据模型调整这个值
    )

    # 应用 PEFT 配置
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc}


    training_args = TrainingArguments(
        output_dir='./checkpoint',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        learning_rate=2e-4,
        save_strategy="no",
        eval_strategy="epoch",
        gradient_accumulation_steps=4,
        remove_unused_columns=True,
        fp16=True,
        load_best_model_at_end=False,  # 由于 save_strategy="no"，这个应该设为 False
        report_to=None,  # 禁用wandb等记录器
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 训练模型
    trainer.train()

    # 预测
    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    # 保存结果
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})

    # 确保输出目录存在
    os.makedirs("./result", exist_ok=True)

    result_output.to_csv("./result/deberta(gpt-2)_prefix.csv", index=False, quoting=3)
    logging.info('result saved!')