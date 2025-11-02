import os
import sys
import logging
import datasets
import pandas as pd
import numpy as np

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    # 最简单稳定的方法：直接使用 sklearn 的 accuracy_score
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(labels, predictions, average='binary')
        recall = recall_score(labels, predictions, average='binary')
        f1 = f1_score(labels, predictions, average='binary')

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

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

    training_logs = trainer.state.log_history
    print("\n=== 训练过程总结 ===")
    for log in training_logs:
        if 'eval_accuracy' in log:
            print(f"Epoch {log.get('epoch', 'N/A')}: 验证准确率 = {log['eval_accuracy']:.4f}")

    # 在验证集上最终评估
    eval_results = trainer.evaluate()
    print(f"\n=== 最终验证结果 ===")
    print(f"验证损失: {eval_results['eval_loss']:.4f}")
    print(f"验证准确率: {eval_results['eval_accuracy']:.4f}")

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(f"预测完成，共 {len(test_pred)} 条预测结果")

    os.makedirs("./result", exist_ok=True)
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/roberta_trainer.csv", index=False, quoting=3)
    logging.info('result saved!')