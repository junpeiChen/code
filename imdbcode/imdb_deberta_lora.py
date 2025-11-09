import os
import sys
import logging
import datasets
import torch

import pandas as pd
import numpy as np

from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train = pd.read_csv(r"/root/autodl-tmp/Downloads/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv(r"/root/autodl-tmp/Downloads/testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train, val = train_test_split(train, test_size=.2)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    model_id = r"/root/autodl-tmp/deberta-v3-small"

    tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = 384  # 适当减少序列长度

    # 修复tokenizer配置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '[PAD]'


    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=384,  # 减少最大长度
            padding=False,
            return_tensors=None
        )


    # 使用多线程预处理数据
    tokenized_train = train_dataset.map(preprocess_function, batched=True, num_proc=4, batch_size=1000)
    tokenized_val = val_dataset.map(preprocess_function, batched=True, num_proc=4, batch_size=1000)
    tokenized_test = test_dataset.map(preprocess_function, batched=True, num_proc=4, batch_size=1000)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 配置8位量化
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["pooler", "classifier"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,  # 使用float16加速计算
    )

    # 对齐tokenizer和model配置
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(tokenizer, 'eos_token_id'):
        model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(tokenizer, 'bos_token_id'):
        model.config.bos_token_id = tokenizer.bos_token_id

    # 优化LoRA配置
    lora_config = LoraConfig(
        r=8,  # 减少秩
        lora_alpha=16,  # 减少alpha
        target_modules=["query_proj", "value_proj"],  # 只针对关键模块，减少计算量
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    # 准备8位模型训练
    model = prepare_model_for_kbit_training(model)

    # 添加LoRA适配器
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc}


    training_args = TrainingArguments(
        output_dir='./checkpoint',
        num_train_epochs=2,  # 减少训练轮数
        per_device_train_batch_size=4,  # 增加批处理大小
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,  # 使用比例而不是固定步数
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,  # 减少日志频率
        save_strategy="no",
        eval_strategy="steps",  # 改为按步数评估
        eval_steps=500,  # 每500步评估一次
        learning_rate=2e-4,
        fp16=True,  # 启用混合精度训练
        dataloader_num_workers=4,  # 增加数据加载工作进程
        dataloader_pin_memory=True,  # 启用内存pin加速数据加载
        gradient_accumulation_steps=2,  # 优化梯度累积
        gradient_checkpointing=True,  # 启用梯度检查点以节省内存
        remove_unused_columns=True,  # 移除未使用的列节省内存
        report_to=[],  # 禁用wandb等报告以节省开销
        optim="adamw_torch",  # 使用torch的优化器
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,  # 修复参数名
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 清空缓存
    torch.cuda.empty_cache()

    trainer.train()

    # 预测时使用更大的批处理大小
    trainer.args.per_device_eval_batch_size = 16  # 增加预测批处理大小
    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/deberta_lora_int8.csv", index=False, quoting=3)
    logging.info('result saved!')