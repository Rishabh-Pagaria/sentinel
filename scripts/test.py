# Test Gemma-2-2b-it model (instruction-tuned, NO fine-tuning)
# Testing if base instruction-tuned model can provide explanations
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Single test email
TEST_EMAIL = {
    "text": """URGENT! You have won $1,000,000 in the lottery! Send us your bank details and social security number to claim your prize NOW!""",
}

print("Loading Gemma-2-2b-it (instruction-tuned base model)...\n")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-2b-it",
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load base instruction-tuned model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "artifacts/gemma-2-2b-it-phishing",
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.eval()

print("Model loaded successfully!\n")
print("="*80)

print(f"\nTest Email: {TEST_EMAIL['text'][:100]}...")
print(f"Expected: {TEST_EMAIL['expected'].upper()}")

# Ask for classification WITH explanation
print(f"\n[Testing: Classification with Explanation]\n")
instruction = "You are a security expert. Analyze the following email and determine if it is 'phishing' or 'safe'. Provide your answer and a brief explanation."

messages = [
    {"role": "user", "content": f"{instruction}\n\nEmail: {TEST_EMAIL['text']}"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f"Prompt:\n{prompt}\n")

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,  # Allow space for explanation
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

input_length = inputs['input_ids'].shape[1]
new_tokens = outputs[0][input_length:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

print(f"Model Response:\n{response}")
print("\n" + "="*80)


