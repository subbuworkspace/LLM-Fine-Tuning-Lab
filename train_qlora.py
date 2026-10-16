"""
QLoRA supervised fine-tuning demo.
Recommended: run in Google Colab with a GPU runtime (T4 or better).
Uses synthetic support-ticket examples; do not treat this tiny dataset as production-ready.
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH = os.getenv("TRAIN_FILE", "data/train.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/support-ticket-qlora")

if not torch.cuda.is_available():
    raise SystemExit(
        "No CUDA GPU detected. Open this project in Google Colab and select "
        "Runtime > Change runtime type > T4 GPU, then rerun."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def to_text(example):
    # Apply the model's own chat template instead of inventing special tokens.
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )}

raw = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = raw.map(to_text, remove_columns=raw.column_names)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    fp16=True,
    max_length=512,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to: {OUTPUT_DIR}")
