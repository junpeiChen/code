import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import logging
import pandas as pd
import numpy as np
import torch
import re
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import DataCollatorForLanguageModeling

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# --- 日志设置 ---
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

# --- 复杂提示词工程 ---
# 训练提示词 - 包含详细的任务描述和示例
train_prompt = """You are an expert sentiment analysis system specialized in movie reviews. 

**TASK**: 
Classify the sentiment of the movie review as either POSITIVE or NEGATIVE.

**GUIDELINES**:
- POSITIVE: Reviews that express satisfaction, enjoyment, or recommendation
- NEGATIVE: Reviews that express disappointment, criticism, or discouragement

**EXAMPLES**:
- "This movie was absolutely fantastic! Great acting and plot." → POSITIVE
- "Terrible movie, wasted my time and money." → NEGATIVE
- "The cinematography was beautiful but the story was weak." → NEGATIVE
- "Despite some flaws, I thoroughly enjoyed this film." → POSITIVE

**REVIEW TO CLASSIFY**:
{}

**SENTIMENT CLASSIFICATION**:
{}"""

# 推理提示词 - 与训练时略有不同，不包含答案
inference_prompt = """You are an expert sentiment analysis system specialized in movie reviews. 

**TASK**: 
Classify the sentiment of the movie review as either POSITIVE or NEGATIVE.

**GUIDELINES**:
- POSITIVE: Reviews that express satisfaction, enjoyment, or recommendation
- NEGATIVE: Reviews that express disappointment, criticism, or discouragement

**EXAMPLES**:
- "This movie was absolutely fantastic! Great acting and plot." → POSITIVE
- "Terrible movie, wasted my time and money." → NEGATIVE
- "The cinematography was beautiful but the story was weak." → NEGATIVE
- "Despite some flaws, I thoroughly enjoyed this film." → POSITIVE

**REVIEW TO CLASSIFY**:
{}

**SENTIMENT CLASSIFICATION**:
"""


# --- 文本预处理函数 ---
def preprocess_text(text):
    """清理和预处理文本"""
    if pd.isna(text):
        return ""

    # 移除HTML标签
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)

    # 移除多余空格和特殊字符
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\\[ntr]', ' ', text)

    # 限制长度
    text = text.strip()[:800]

    return text


# --- 数据格式化函数 ---
def formatting_prompts_func(examples):
    inputs = examples["text"]
    labels = examples["label"]
    outputs_text = []

    for input_text, label in zip(inputs, labels):
        # 预处理文本
        clean_text = preprocess_text(input_text)
        label_text = "POSITIVE" if label == 1 else "NEGATIVE"
        text = train_prompt.format(clean_text, label_text) + tokenizer.eos_token
        outputs_text.append(text)

    return {"text": outputs_text}


# --- 高级解析函数 ---
def parse_model_output(generated_text):
    """高级解析模型输出，使用多种策略"""
    generated_text_lower = generated_text.lower()

    # 策略1: 查找分类标记后的内容
    classification_markers = ["sentiment classification:", "classification:", "sentiment:"]

    for marker in classification_markers:
        if marker in generated_text_lower:
            parts = generated_text_lower.split(marker, 1)
            if len(parts) > 1:
                response_part = parts[1].strip()
                # 提取第一个单词或前几个单词
                first_word = response_part.split()[0] if response_part.split() else ""

                if first_word in ["positive", "pos"]:
                    return 1
                elif first_word in ["negative", "neg"]:
                    return 0

    # 策略2: 直接搜索关键词（带上下文）
    positive_indicators = [
        "positive", "pos", "good", "great", "excellent", "amazing",
        "wonderful", "fantastic", "brilliant", "love", "liked", "enjoyed",
        "recommend", "awesome", "outstanding"
    ]

    negative_indicators = [
        "negative", "neg", "bad", "terrible", "awful", "horrible",
        "boring", "waste", "disappointing", "hate", "dislike", "poor",
        "worst", "weak", "terrible"
    ]

    # 检查整个生成文本中的关键词
    positive_count = sum(1 for word in positive_indicators if word in generated_text_lower)
    negative_count = sum(1 for word in negative_indicators if word in generated_text_lower)

    if positive_count > negative_count:
        return 1
    elif negative_count > positive_count:
        return 0

    # 策略3: 使用正则表达式匹配模式
    positive_patterns = [
        r'\bpositive\b', r'\bpos\b', r'classify.*positive', r'sentiment.*positive'
    ]
    negative_patterns = [
        r'\bnegative\b', r'\bneg\b', r'classify.*negative', r'sentiment.*negative'
    ]

    for pattern in positive_patterns:
        if re.search(pattern, generated_text_lower):
            return 1

    for pattern in negative_patterns:
        if re.search(pattern, generated_text_lower):
            return 0

    # 策略4: 基于情感词汇的启发式分析
    strong_positive_words = ["love", "amazing", "fantastic", "brilliant", "masterpiece"]
    strong_negative_words = ["hate", "terrible", "awful", "horrible", "worst"]

    for word in strong_positive_words:
        if word in generated_text_lower:
            return 1

    for word in strong_negative_words:
        if word in generated_text_lower:
            return 0

    # 默认返回负向（保守策略）
    return 0


# --- 主执行程序 ---
if __name__ == '__main__':
    logger.info(r"running %s" % ''.join(sys.argv))

    # --- 1. 加载和准备数据 ---
    logger.info("Loading data...")
    try:
        train_df = pd.read_csv(r"/root/autodl-tmp/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
        test_df = pd.read_csv(r"/root/autodl-tmp/testData.tsv", header=0, delimiter="\t", quoting=3)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    # 数据分析和统计
    logger.info(f"Training data size: {len(train_df)}")
    logger.info(f"Sentiment distribution: {train_df['sentiment'].value_counts().to_dict()}")

    # 分析评论长度
    train_df['review_length'] = train_df['review'].str.len()
    logger.info(f"Review length stats - Mean: {train_df['review_length'].mean():.2f}, "
                f"Max: {train_df['review_length'].max()}, Min: {train_df['review_length'].min()}")

    # 分割训练和验证集
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        random_state=3407,
        stratify=train_df['sentiment']
    )

    # 创建数据集
    train_dataset = Dataset.from_dict({'label': train_df["sentiment"], 'text': train_df['review']})
    val_dataset = Dataset.from_dict({'label': val_df["sentiment"], 'text': val_df['review']})
    test_dataset = Dataset.from_dict({"text": test_df['review']})

    # --- 2. 加载模型和 Tokenizer ---
    logger.info("Loading Qwen model...")
    model_name = r"/root/autodl-tmp/Qwen2.5-0.5B-Instruct"

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    # --- 3. PEFT (LoRA) 设置 ---
    logger.info("Setting up PEFT...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        random_state=3407,
        use_gradient_checkpointing="unsloth",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # --- 4. 格式化数据集 ---
    logger.info("Formatting datasets with advanced prompts...")
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    # --- 5. 训练参数 ---
    logger.info("Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir="./qwen_advanced_prompt_output",
        overwrite_output_dir=True,
        per_device_train_batch_size=4,  # 较小的批次大小以适应更长的序列
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,  # 使用比例而非固定步数
        num_train_epochs=4,
        learning_rate=3e-5,  # 稍低的学习率
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=3407,
        data_seed=3407,
        group_by_length=True,  # 按长度分组提高效率
        dataloader_pin_memory=False,
    )

    # --- 6. 初始化 SFTTrainer ---
    logger.info("Initializing SFTTrainer with advanced prompts...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # --- 7. 训练 ---
    logger.info("Starting training with advanced prompts...")

    # 训练前评估
    logger.info("Pre-training evaluation...")
    pre_train_eval = trainer.evaluate()
    logger.info(f"Pre-training evaluation results: {pre_train_eval}")

    # 训练模型
    train_result = trainer.train()

    # 保存最终模型
    trainer.save_model()
    logger.info("Model saved successfully.")

    # 训练后评估
    logger.info("Post-training evaluation...")
    post_train_eval = trainer.evaluate()
    logger.info(f"Post-training evaluation results: {post_train_eval}")

    # 记录训练指标
    metrics = train_result.metrics
    logger.info(f"Training metrics: {metrics}")

    # --- 8. 在验证集上评估准确率 ---
    logger.info("Evaluating on validation set with advanced parsing...")
    FastLanguageModel.for_inference(model)

    val_texts = val_df['review'].tolist()
    val_labels = val_df['sentiment'].tolist()
    val_predictions = []

    # 使用批次推理
    batch_size = 8  # 较小的批次大小以适应更长的提示词

    for i in range(0, len(val_texts), batch_size):
        batch_texts = val_texts[i:i + batch_size]
        batch_prompts = [inference_prompt.format(preprocess_text(text)) for text in batch_texts]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(
            "cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,  # 增加生成长度以容纳更复杂的响应
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1,
            )

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, generated_text in enumerate(generated_texts):
            prediction = parse_model_output(generated_text)
            val_predictions.append(prediction)

            # 记录一些示例用于调试
            if i == 0 and j < 3:
                logger.info(f"Example {j + 1}:")
                logger.info(f"Prompt: {batch_prompts[j][:200]}...")
                logger.info(f"Generated: {generated_text}")
                logger.info(f"Predicted: {prediction}, Actual: {val_labels[i + j]}")

    # 计算验证集准确率
    val_accuracy = accuracy_score(val_labels, val_predictions)
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

    # 详细分类报告
    logger.info(f"Classification Report:\n{classification_report(val_labels, val_predictions)}")

    # 混淆矩阵
    cm = confusion_matrix(val_labels, val_predictions)
    logger.info(f"Confusion Matrix:\n{cm}")

    # --- 9. 错误分析 ---
    logger.info("Performing error analysis...")
    errors = []
    for i, (true, pred) in enumerate(zip(val_labels, val_predictions)):
        if true != pred:
            errors.append({
                'text': val_texts[i][:500] + "..." if len(val_texts[i]) > 500 else val_texts[i],
                'true': true,
                'predicted': pred
            })

    logger.info(f"Number of errors: {len(errors)}")
    if errors:
        logger.info("Sample errors:")
        for i, error in enumerate(errors[:3]):
            logger.info(f"Error {i + 1}: True={error['true']}, Predicted={error['predicted']}")
            logger.info(f"Text: {error['text']}")

    # --- 10. 在测试集上进行预测 ---
    logger.info("Starting test set inference with advanced prompts...")

    test_texts = test_df['review'].tolist()
    test_ids = test_df['id'].tolist()
    test_predictions = []

    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i:i + batch_size]
        batch_prompts = [inference_prompt.format(preprocess_text(text)) for text in batch_texts]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(
            "cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1,
            )

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for generated_text in generated_texts:
            prediction = parse_model_output(generated_text)
            test_predictions.append(prediction)

        if (i // batch_size) % 20 == 0:
            logger.info(f"Processed {i + len(batch_texts)}/{len(test_texts)} test samples")

    # --- 11. 保存结果 ---
    logger.info("Saving results...")
    results_dir = "./advanced_prompt_results"
    os.makedirs(results_dir, exist_ok=True)

    # 保存测试集预测
    result_output = pd.DataFrame(data={"id": test_ids, "sentiment": test_predictions})
    result_path = os.path.join(results_dir, "qwen_advanced_prompt_predictions.csv")
    result_output.to_csv(result_path, index=False, quoting=3)
    logger.info(f'Test predictions saved to: {result_path}')

    # 保存验证集结果用于分析
    val_results = pd.DataFrame({
        "review": val_texts,
        "true_label": val_labels,
        "predicted_label": val_predictions
    })
    val_results_path = os.path.join(results_dir, "advanced_validation_results.csv")
    val_results.to_csv(val_results_path, index=False)
    logger.info(f'Validation results saved to: {val_results_path}')

    # 保存错误分析
    if errors:
        errors_df = pd.DataFrame(errors)
        errors_path = os.path.join(results_dir, "error_analysis.csv")
        errors_df.to_csv(errors_path, index=False)
        logger.info(f'Error analysis saved to: {errors_path}')

    # 最终统计
    logger.info("=== FINAL RESULTS ===")
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    logger.info(
        f"Test set predictions - Positive: {sum(test_predictions)}, Negative: {len(test_predictions) - sum(test_predictions)}")

    if val_accuracy >= 0.80:
        logger.info("🎉 SUCCESS: Model achieved target accuracy of 80% or higher!")
    elif val_accuracy >= 0.75:
        logger.info("✅ GOOD: Model achieved good accuracy (75-80%).")
    else:
        logger.info("⚠️  NEEDS IMPROVEMENT: Model accuracy below 75%. Consider:")
        logger.info("   - Increasing training epochs")
        logger.info("   - Trying a larger model")
        logger.info("   - Further prompt engineering")
        logger.info("   - Hyperparameter tuning")