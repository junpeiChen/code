import os
import sys
import logging
import datasets
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from peft import PromptEncoderConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

# 加载数据
train = pd.read_csv(r"/root/autodl-tmp/Downloads/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv(r"/root/autodl-tmp/Downloads/testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # 数据集划分
    train, val = train_test_split(train, test_size=.2)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    batch_size = 32  # 可以减小batch size以避免GPU内存溢出

    model_id = r"/root/autodl-tmp/deberta-v3-small"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)

    # 数据预处理
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    # ptuning 配置
    peft_config = PromptEncoderConfig(
        num_virtual_tokens=20,  # 可以调整虚拟token的数量
        encoder_hidden_size=128,
        task_type=TaskType.SEQ_CLS
    )

    # 准备8位量化模型
    # model = prepare_model_for_kbit_training(model)

    # 加入LoRA适配器
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 计算准确率
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc}

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=4,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=50,
        learning_rate=2e-4,
        save_strategy="no",  # 每个epoch保存
        eval_strategy="epoch",  # 每个epoch进行评估
        gradient_accumulation_steps=4,  # 梯度累积，增加训练稳定性
        remove_unused_columns=True,     # 移除未使用的列
        fp16=True,  # 启用混合精度训练，减少显存占用
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # 直接使用函数
    )

    # 开始训练
    trainer.train()

    # 预测
    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    # 保存结果
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/deberta_ptuning_int8.csv", index=False, quoting=3)
    logging.info('result saved!')
