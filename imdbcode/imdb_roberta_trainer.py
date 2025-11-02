import os
import sys
import logging
import datasets
import pandas as pd
import numpy as np
import evaluate

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

LOCAL_MODEL_PATH = r"/root/autodl-tmp/roberta-base"

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

    tokenizer = RobertaTokenizerFast.from_pretrained(LOCAL_MODEL_PATH)

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = RobertaForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)

    # 修复：使用正确的 evaluate 加载方式
    try:
        # 方法1：直接使用 evaluate 库
        accuracy_metric = evaluate.load("accuracy")
    except Exception as e:
        print(f"使用 evaluate 库失败: {e}")
        # 方法2：手动实现 accuracy 计算
        def compute_accuracy(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {"accuracy": (predictions == labels).mean()}
        compute_metrics = compute_accuracy
    else:
        # 方法1成功的情况
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return accuracy_metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=5e-6,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()

    os.makedirs("./result", exist_ok=True)
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/roberta_trainer.csv", index=False, quoting=3)
    logging.info('result saved!')