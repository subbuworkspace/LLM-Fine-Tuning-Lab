"""Load a base model plus a trained LoRA adapter and classify one ticket."""
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "outputs/support-ticket-qlora"

parser = argparse.ArgumentParser()
parser.add_argument("--ticket", required=True, help="Support ticket text")
parser.add_argument("--adapter", default=ADAPTER_DIR, help="Path to trained adapter")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, args.adapter)
model.eval()

prompt = (
    "Classify this support ticket into one category: account_access, billing, "
    "bug_report, feature_request. Return only the category. Ticket: " + args.ticket
)
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
new_tokens = output[0][input_ids.shape[-1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
