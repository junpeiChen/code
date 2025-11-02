import os
import sys
import logging
import datasets
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 定义本地模型路径
#local_model_path = r"D:\Pycharm\bert-base-uncased"
#服务器下地址
LOCAL_BERT_PATH = r"/root/autodl-tmp/bert-base-uncased"

# 设置环境变量，强制离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

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

    # 强制使用本地分词器
    try:
        tokenizer = BertTokenizerFast.from_pretrained(
            LOCAL_BERT_PATH,
            local_files_only=True,
            force_download=False
        )
        logger.info("成功加载本地分词器")
    except Exception as e:
        logger.error(f"加载本地分词器失败: {e}")
        logger.info("请检查本地模型路径是否正确，文件是否完整")
        sys.exit(1)

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 强制使用本地模型
    try:
        model = BertForSequenceClassification.from_pretrained(
            LOCAL_BERT_PATH,
            local_files_only=True,
            force_download=False,
            num_labels=2  # 二分类任务
        )
        logger.info("成功加载本地模型")
    except Exception as e:
        logger.error(f"加载本地模型失败: {e}")
        logger.info("请检查本地模型文件是否完整")
        sys.exit(1)

    # 使用 sklearn 计算指标
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

    # 兼容新旧版本的 TrainingArguments
    try:
        training_args = TrainingArguments(
            output_dir='./checkpoint',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            save_strategy="no",
            eval_strategy="epoch"
        )
    except TypeError:
        # 如果新参数名不支持，使用旧参数名
        training_args = TrainingArguments(
            output_dir='./checkpoint',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            save_strategy="no",
            evaluation_strategy="epoch"
        )

    # 兼容新旧版本的 Trainer 参数
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            processing_class=tokenizer,  # 使用新的参数名
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        logger.info("使用 processing_class 初始化 Trainer")
    except TypeError:
        # 如果新参数名不支持，回退到旧参数名
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,  # 回退到旧参数名
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        logger.info("使用 tokenizer 参数初始化 Trainer")

    trainer.train()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/bert_trainer.csv", index=False, quoting=3)
    logging.info('result saved!')