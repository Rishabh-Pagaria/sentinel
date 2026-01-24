# Quick debug script to see what the model is actually generating

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-2b-it",
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "hf_models/gemma-2-2b",
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Load LoRA adapter and MERGE it
model = PeftModel.from_pretrained(model, "artifacts/models/gemma-2-2b-phishing/new-checkpoint")
model = model.merge_and_unload()  # Merge LoRA weights into base model
model.eval()

print(f"\nModel type: {type(model)}")
print(f"Model merged: LoRA weights integrated into base model")
print()

# Load a few samples
dataset = load_dataset("json", data_files="data/eval_gemma.jsonl", split="train")

print("Testing first 5 samples...\n")
print("="*80)

for i in range(5):
    sample = dataset[i]
    
    print(f"\n--- Sample {i+1} ---")
    print(f"Expected: {sample['output']}")
    print(f"Input (first 100 chars): {sample['input'][:100]}...")
    
    # Method 1: Chat template with add_generation_prompt
    messages = [
        {"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"\nPrompt format:\n{prompt[:200]}...")
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0,
        )
    
    print(f"Output token IDs: {outputs[0].tolist()[-10:]}")  # Last 10 tokens
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = generated[len(prompt):].strip()
    
    # Also decode just the new tokens
    input_length = inputs['input_ids'].shape[1]
    new_tokens = outputs[0][input_length:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    
    print(f"\nGenerated (full): '{prediction}'")
    print(f"New tokens only: {new_tokens.tolist()}")
    print(f"New tokens decoded: '{new_text}'")
    print("="*80)
