"""
Convert training data to Gemma-2 chat template format using official tokenizer method.
This ensures the model learns with the correct conversation structure.
"""

import json
import sys
from transformers import AutoTokenizer

def convert_to_gemma_format(input_file, output_file, tokenizer):
    """
    Convert JSON lines with instruction/input/output to Gemma chat template format.
    Uses tokenizer.apply_chat_template() for official format compliance.
    """
    
    converted_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract fields
                instruction = data.get('instruction', '').strip()
                email_input = data.get('input', '').strip()
                output = data.get('output', '').strip()
                
                # Combine instruction and input for the user message
                user_message = f"{instruction}\n\n{email_input}"
                
                # Use official tokenizer.apply_chat_template() method
                # This handles <bos>, spacing, and all special tokens correctly
                chat = [
                    {"role": "user", "content": user_message},
                    {"role": "model", "content": output}
                ]
                formatted_text = tokenizer.apply_chat_template(
                    chat, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                # Store in new JSON format that's compatible with SFTTrainer
                output_data = {
                    "text": formatted_text,
                    "instruction": instruction,
                    "input": email_input,
                    "output": output
                }
                
                outfile.write(json.dumps(output_data) + '\n')
                converted_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1
    
    print(f"\n✓ Conversion complete!")
    print(f"  Converted: {converted_count} samples")
    print(f"  Errors: {error_count} samples")
    print(f"  Output file: {output_file}")

if __name__ == "__main__":
    base_path = r"c:\Users\rpaga\Desktop\sentinel\sentinel\data"
    model_path = r"c:\Users\rpaga\Desktop\sentinel\sentinel\hf_models\gemma-2-2b-it"
    
    # Load tokenizer (has built-in chat template for Gemma-2)
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        sys.exit(1)
    
    files_to_convert = [
        ("train_gemma_clean.jsonl", "train_gemma_template.jsonl"),
        ("eval_gemma_clean.jsonl", "eval_gemma_template.jsonl"),
        ("test_gemma_clean.jsonl", "test_gemma_template.jsonl"),
    ]
    
    print("=" * 70)
    print("CONVERTING ALL DATASET FILES TO GEMMA-2 CHAT TEMPLATE FORMAT")
    print("=" * 70)
    
    for input_file, output_file in files_to_convert:
        input_path = f"{base_path}\\{input_file}"
        output_path = f"{base_path}\\{output_file}"
        
        print(f"\n[{input_file}]")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print("-" * 70)
        
        try:
            convert_to_gemma_format(input_path, output_path, tokenizer)
        except FileNotFoundError:
            print(f"  ⚠️  WARNING: Input file not found - skipping")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print("\n" + "=" * 70)
    print("ALL CONVERSIONS COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Verify template files were created successfully")
    print("2. Run training with train_gemma_lora.py")
    print("=" * 70)
