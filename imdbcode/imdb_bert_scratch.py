import os
import sys
import logging
import datasets
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

# 设置本地模型路径
LOCAL_BERT_PATH = r"/root/autodl-tmp/bert-base-uncased"

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

train = pd.read_csv(r"/root/autodl-tmp/Downloads/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv(r"/root/autodl-tmp/Downloads/testData.tsv", header=0, delimiter="\t", quoting=3)


class BertScratch(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # 设备信息
    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    train, val = train_test_split(train, test_size=.2)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    # 加载分词器
    try:
        tokenizer = BertTokenizerFast.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)
        logger.info("成功加载本地分词器")
    except Exception as e:
        logger.error(f"加载本地分词器失败: {e}")
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)


    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 加载模型并移动到设备
    try:
        model = BertScratch.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)
        model.to(device)
        logger.info("成功加载本地模型并移动到设备")
    except Exception as e:
        logger.error(f"加载本地模型失败: {e}")
        from transformers import BertConfig

        config = BertConfig.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)
        config.num_labels = 2
        model = BertScratch(config)
        model.to(device)
        logger.info("使用随机初始化的模型并移动到设备")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}


    training_args = TrainingArguments(
        output_dir='./checkpoint',
        num_train_epochs=3,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=12,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="no",
        eval_strategy="epoch",
        load_best_model_at_end=False,
        no_cuda=not torch.cuda.is_available(),
    )

    # 初始化 Trainer
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        logger.info("使用 processing_class 初始化 Trainer")
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        logger.info("使用 tokenizer 参数初始化 Trainer")

    # 开始训练
    logger.info("开始训练...")
    trainer.train()

    # 预测时移除标签列（如果存在）
    if 'label' in tokenized_test.column_names:
        tokenized_test = tokenized_test.remove_columns(['label'])

    logger.info("开始预测...")
    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs.predictions, axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/bert_scratch.csv", index=False, quoting=3)
    logger.info('结果已保存!')