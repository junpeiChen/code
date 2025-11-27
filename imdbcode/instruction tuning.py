# 导入库
import os
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. 登录HuggingFace
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("Gemma")
login(hf_token)

# 2. 加载模型和tokenizer
print("正在加载模型...")
max_seq_length = 2048
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
    token=hf_token,
)

# 3. 定义提示模板
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
{}"""

# 4. 微调前推理测试
print("=== 微调前推理测试 ===")
question = "A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings, what would cystometry most likely reveal about her residual volume and detrusor contractions?"

FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=500,
    use_cache=True,
)

response_before = tokenizer.batch_decode(outputs)[0].split("### Response:")[1]
print("微调前回答:", response_before[:500] + "...")

# 5. 加载训练数据
print("正在加载训练数据...")
try:
    dataset = load_dataset(
        "FreedomIntelligence/medical-01-reasoning-SFT",
        "en",
        split="train[0:100]",  # 先用100个样本测试
        trust_remote_code=True,
        token=hf_token
    )
except Exception as e:
    print(f"数据集加载失败: {e}")
    print("使用示例数据替代...")
    dataset = Dataset.from_dict({
        "Question": [question] * 10,
        "Complex_CoT": ["思考过程示例..."] * 10,
        "Response": ["回答示例..."] * 10
    })

# 6. 数据预处理
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples.get("Complex_CoT", [""] * len(inputs))
    outputs = examples.get("Response", [""] * len(inputs))
    texts = []
    for i, (input, cot, output) in enumerate(zip(inputs, cots, outputs)):
        if not cot.strip():
            cot = "这是一个医学推理过程..."
        text = prompt_style.format(input, f"<think>\n{cot}\n</think>\n{output}") + EOS_TOKEN
        texts.append(text)
    # 确保所有文本都是字符串，防止 int 转换报错
    texts = [str(t) for t in texts]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# 7. 配置LoRA进行微调
print("配置LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 8. 设置训练参数
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to=[],
    ),
)

# 9. 开始训练
print("开始训练...")
trainer_stats = trainer.train()
print("训练完成!")

# 10. 微调后推理测试
print("=== 微调后推理测试 ===")
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=500,
    use_cache=True,
)

response_after = tokenizer.batch_decode(outputs)[0].split("### Response:")[1]
print("微调后回答:", response_after[:500] + "...")

# 11. 保存模型
print("保存模型...")
model.save_pretrained("DeepSeek-R1-Medical-COT")
tokenizer.save_pretrained("DeepSeek-R1-Medical-COT")

print("=== 训练完成 ===")
