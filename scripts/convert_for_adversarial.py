#!/usr/bin/env python3
"""
Prepare training data for adversarial example generation.
Converts from our format to the format expected by generate_adversarial.py
"""

import json
import os
import sys
from typing import Dict, List

def convert_example(example: Dict) -> Dict:
    """Convert from our format to adversarial generation format"""
    text = example.get('input', '')
    if text.startswith('EMAIL: '):
        text = text[7:]  # Remove EMAIL: prefix
        
    # Skip empty examples
    if not text.strip() or text.strip() == 'empty':
        return None
        
    label = example.get('output', {}).get('label')
    if not label:
        return None
        
    return {
        'text': text,
        'true_label': label,
        'pred_label': label,  # Use true label as prediction for training data
        'confidence': example.get('output', {}).get('confidence', 1.0),
        'tactics': example.get('output', {}).get('tactics', []),
    }

def main():
    # Read training data
    input_file = 'out_jsonl/train.jsonl'
    temp_file = 'out_jsonl/train_converted.jsonl'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        sys.exit(1)
    
    # Convert examples
    converted = []
    with open(input_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            converted_example = convert_example(example)
            if converted_example:  # Skip None results
                converted.append(converted_example)
    
    print(f"Converted {len(converted)} valid examples")
    
    # Save converted examples
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    with open(temp_file, 'w') as f:
        for example in converted:
            f.write(json.dumps(example) + '\n')
    
    print(f"Saved converted examples to {temp_file}")
    print("\nNow run:")
    print(f"python scripts/generate_adversarial.py --source-file {temp_file} --num-examples 200 --no-backtranslate")

if __name__ == '__main__':
    main()