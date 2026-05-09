import json
import sys
import os
sys.path.append(os.getcwd())

from src.metrics import compute_metrics
from src.strip_thinking import extract_code

def verify():
    with open('data/thinkanywhere_sft_v4.jsonl', 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    
    print(f"Verifying {len(examples)} examples from Dataset v4...")
    
    loops = 0
    lazy = 0
    total = len(examples)
    
    for ex in examples:
        raw_output = ex['raw_output']
        clean_code, _ = extract_code(raw_output)
        
        # Test result is empty as we only care about metrics
        m = compute_metrics(clean_code, raw_output, {})
        
        if m['has_thinking_loop']:
            loops += 1
        if m['is_lazy']:
            lazy += 1
            
    print(f"Summary:")
    print(f"- Total: {total}")
    print(f"- Loops: {loops}")
    print(f"- Lazy: {lazy}")

if __name__ == "__main__":
    verify()
