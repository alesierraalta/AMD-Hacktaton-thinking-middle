import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

def probe_adapter(base_model_name, adapter_path, prompt):
    print(f"--- Adapter Probe ---")
    print(f"Base Model: {base_model_name}")
    print(f"Adapter Path: {adapter_path}")
    
    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter path {adapter_path} does not exist.")
        return False

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Load Base Model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Base Output
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"\nPrompt: {prompt}")
    
    print("\n--- Base Model Output ---")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
        base_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(base_text)

    # Load Adapter
    print("\nAttaching PEFT adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    print("\n--- Adapter Output ---")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
        adapter_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(adapter_text)

    # Comparison
    diff = base_text != adapter_text
    print(f"\nOutputs differ: {'YES' if diff else 'NO'}")
    
    # Cleanup to avoid OOM
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return diff

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen1.5-1.8B-Chat")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Write a python function to add two numbers.")
    args = parser.parse_args()
    
    probe_adapter(args.base_model, args.adapter_path, args.prompt)
