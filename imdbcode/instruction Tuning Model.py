
# ===============================
# 1. 导入库
# ===============================
import torch, json
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ===============================
# 2. 设置 Hugging Face token（用于 gated repo）
# ===============================
HF_TOKEN = ""  # 替换为你自己的 Hugging Face token

# ===============================
# 3. 使用大模型生成 SST-2 合成数据
# ===============================
big_model_name = "google/gemma-7b"  # gated repo
big_tokenizer = AutoTokenizer.from_pretrained(big_model_name, use_auth_token=HF_TOKEN)
big_model = AutoModelForCausalLM.from_pretrained(
    big_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN
)
big_model.gradient_checkpointing_enable()

labels = ["positive", "negative"]
synthetic_data = []
prompt_template = "Generate a short movie review sentence labeled as {label} (positive/negative):"

for label in labels:
    for i in range(2000):  # 每类 2000 条
        prompt = prompt_template.format(label=label)
        inputs = big_tokenizer(prompt, return_tensors="pt").to(device)
        outputs = big_model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        text = big_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        synthetic_data.append({"text": text, "label": 1 if label == "positive" else 0})

with open("synthetic_sst2.json", "w") as f:
    for item in synthetic_data:
        f.write(json.dumps(item) + "\n")

print("合成数据生成完成，共计:", len(synthetic_data))

# ===============================
# 4. 数据集格式化为 Instruction-Response
# ===============================
dataset = Dataset.from_json("synthetic_sst2.json", split="train")


def format_for_sft(example):
    instruction = "Determine the sentiment of the following sentence and answer with 'positive' or 'negative'."
    input_text = example["text"]
    output_text = "positive" if example["label"] == 1 else "negative"
    return {"text": f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n{output_text}"}


dataset = dataset.map(format_for_sft)

# ===============================
# 5. 定义小模型列表（3B 参数左右）
# ===============================
small_models = {
    "llama": "meta-llama/Llama-3.2-3B",                         # 3B参数
    "qwen": "Qwen/Qwen2.5-1.5B",                                # 1.5B参数
    "gemma": "google/gemma-2-2b-it",                            # 2B参数
    "phi": "microsoft/phi-2",                                   # 3B参数
    "Mistral": "mistralai/Voxtral-Mini-3B-2507",                # 3B参数
}

results = {}

# ===============================
# 6. 遍历微调模型
# ===============================
for model_name in small_models:
    print(f"\n=== 开始微调模型: {model_name} ===\n")

    # -------------------------------
    # 加载 Tokenizer
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------
    # 加载模型 + 4-bit 量化
    # -------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        token=HF_TOKEN
    )

    # -------------------------------
    # LoRA 配置（必须，否则 4-bit 会报错）
    # -------------------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------------
    # 微调
    # -------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,  # TRL 0.25 必须提供
    train_dataset=dataset,
    dataset_text_field="text",  # 数据字段名(如果你的字段是别的就改这里)
    max_seq_length=256,
    packing=False,  # 如果不启用packing必须写明
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=60,
        learning_rate=2e-4,
        fp16=True,  # 可以保持不变
        logging_steps=5,
        optim="adamw_8bit",
        output_dir="./gemma_sst2_sft",
        save_strategy="steps",
        save_steps=30,
        report_to=[]
    )
)

trainer.train()

# ===============================
# 7. SST-2 验证集评估
# ===============================
print(f"=== 开始评估模型: {model_name} ===")
sst2_val = load_dataset("glue", "sst2", split="validation")


def predict_sentiment(example):
    prompt = f"Determine the sentiment of the following sentence and answer with 'positive' or 'negative':\n{example['sentence']}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    return 1 if "positive" in text else 0


sst2_val = sst2_val.map(lambda x: {"pred": predict_sentiment(x)}, batched=False)
acc = accuracy_score(sst2_val["label"], sst2_val["pred"])
results[model_name] = acc

# ===============================
# 8. 保存微调模型
# ===============================
model.save_pretrained(f"./sst2_small_lora_{model_name.split('/')[-1]}")
tokenizer.save_pretrained(f"./sst2_small_lora_{model_name.split('/')[-1]}")
print(f"=== 模型 {model_name} 保存完成 ===")

# ===============================
# 9. 打印所有小模型的 SST-2 指令学习准确度
# ===============================
print("\n=== SST-2 指令学习准确度 ===")
for k, v in results.items():
    print(k, ":", v)
