import os

# 设置Hugging Face缓存路径
cache_path = "D:\huggingface_cache"
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_path, 'models')
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_path, 'datasets')
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(cache_path, 'hub')

# 确保目录存在
for path in [
    os.environ['TRANSFORMERS_CACHE'],
    os.environ['HF_DATASETS_CACHE'],
    os.environ['HUGGINGFACE_HUB_CACHE']
]:
    os.makedirs(path, exist_ok=True)
from huggingface_hub import login

login(token="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")

# 设置缓存路径
os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache/models'
for path in ['D:/huggingface_cache', 'D:/huggingface_cache/models']:
    os.makedirs(path, exist_ok=True)

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
from tqdm import tqdm

# 配置4-bit量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 测试模型列表
MODELS = {
    "Gemma-2B": "google/gemma-2-2b-it",
    "Qwen-0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-2-2.7B": "microsoft/phi-2",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
}

# 加载数据集
print("加载SST-2数据集...")
dataset = load_dataset("glue", "sst2")
test_dataset = dataset["validation"]
print(f"测试集大小: {len(test_dataset)}")


def create_prompt(model_name, text):
    """为不同模型创建合适的提示词"""
    base_instruction = f"""请分析以下文本的情感倾向。文本来自在线评论，请判断情感是正面还是负面。

文本: "{text}"

请只回复一个数字：
- 如果情感是正面的，回复 1
- 如果情感是负面的，回复 0

回复:"""

    if "gemma" in model_name.lower():
        return f"<start_of_turn>user\n{base_instruction}<end_of_turn>\n<start_of_turn>model\n"
    elif "qwen" in model_name.lower():
        return f"<|im_start|>user\n{base_instruction}<|im_end|>\n<|im_start|>assistant\n"
    elif "phi" in model_name.lower():
        return f"<|user|>\n{base_instruction}<|end|>\n<|assistant|>\n"
    elif "llama" in model_name.lower():
        return f"<|user|>\n{base_instruction}</s>\n<|assistant|>\n"
    else:
        return base_instruction


def extract_sentiment(text):
    """从模型输出中提取0或1的情感标签"""
    # 方法1: 直接查找数字
    numbers = re.findall(r'\b[01]\b', text)
    if numbers:
        return int(numbers[0])

    # 方法2: 关键词匹配
    text_lower = text.lower()
    positive_words = ['positive', 'pos', '正面', '好评', '好', '1']
    negative_words = ['negative', 'neg', '负面', '差评', '差', '0']

    if any(word in text_lower for word in positive_words):
        return 1
    elif any(word in text_lower for word in negative_words):
        return 0

    return None


def evaluate_model_accuracy(model_name, model_id, dataset, sample_size=200):
    """评估单个模型在SST-2上的准确率"""
    print(f"\n评估模型: {model_name}")

    try:
        # 加载模型
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # 选择样本
        eval_dataset = dataset.shuffle(seed=42).select(range(sample_size))

        correct = 0
        total_valid = 0

        print(f"正在评估 {sample_size} 个样本...")

        for i, item in enumerate(tqdm(eval_dataset)):
            text = item['sentence']
            true_label = item['label']

            prompt = create_prompt(model_name, text)

            try:
                # 生成响应
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=tokenizer.eos_token_id
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=False)
                model_response = response[len(prompt):].strip()

                pred = extract_sentiment(model_response)

                if pred is not None:
                    total_valid += 1
                    if pred == true_label:
                        correct += 1

            except Exception as e:
                continue  # 跳过错误样本

        # 计算准确率
        accuracy = correct / total_valid if total_valid > 0 else 0.0
        valid_ratio = total_valid / sample_size

        print(f"{model_name} 结果:")
        print(f"  样本数: {sample_size}")
        print(f"  有效预测: {total_valid} ({valid_ratio:.2%})")
        print(f"  正确预测: {correct}")
        print(f"  准确率: {accuracy:.4f}")

        # 清理内存
        del model, tokenizer
        torch.cuda.empty_cache()

        return {
            "model": model_name,
            "accuracy": accuracy,
            "valid_predictions": total_valid,
            "total_predictions": sample_size,
            "valid_ratio": valid_ratio,
            "correct_predictions": correct
        }

    except Exception as e:
        print(f"评估 {model_name} 时出错: {e}")
        return {
            "model": model_name,
            "accuracy": 0.0,
            "valid_predictions": 0,
            "total_predictions": sample_size,
            "valid_ratio": 0.0,
            "correct_predictions": 0,
            "error": str(e)
        }


def benchmark_all_models(models_dict, dataset, sample_size=200):
    """评估所有模型并比较准确率"""
    print("=" * 60)
    print("SST-2 指令学习准确率基准测试")
    print(f"样本数量: {sample_size}")
    print("=" * 60)

    results = []

    for model_name, model_id in models_dict.items():
        result = evaluate_model_accuracy(model_name, model_id, dataset, sample_size)
        results.append(result)

    # 按准确率排序并显示结果
    print("\n" + "=" * 80)
    print("准确率排名")
    print("=" * 80)

    sorted_results = sorted([r for r in results if r['valid_predictions'] > 0],
                            key=lambda x: x['accuracy'], reverse=True)

    for i, result in enumerate(sorted_results):
        print(f"{i + 1}. {result['model']}: {result['accuracy']:.4f} "
              f"(有效样本: {result['valid_predictions']}/{result['total_predictions']})")

    # 显示无法评估的模型
    failed_models = [r for r in results if r['valid_predictions'] == 0]
    if failed_models:
        print(f"\n无法评估的模型:")
        for model in failed_models:
            print(f"  {model['model']}: {model.get('error', '未知错误')}")

    return sorted_results


# 运行测试
if __name__ == "__main__":
    sample_size = 200

    print("开始SST-2指令学习准确率测试...")
    results = benchmark_all_models(MODELS, test_dataset, sample_size)

    # 简单可视化结果
    if results:
        print("\n准确率对比:")
        for result in results:
            bar_length = int(result['accuracy'] * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"{result['model']:15} {bar} {result['accuracy']:.4f}")