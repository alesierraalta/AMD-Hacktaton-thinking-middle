import argparse
import os
import json
import random

def validate_code(code: str) -> bool:
    placeholders = ["pass", "# implementation", "# todo"]
    code_lower = code.lower()
    for p in placeholders:
        if p in code_lower:
            return False
    return True

def generate_thinking_block() -> str:
    tags = [
        ("thinkanywhere", "/thinkanywhere"),
        ("thinking", "/thinking"),
        ("thought", "/thought"),
        ("reasoning", "/reasoning")
    ]
    headers = [
        "Analysis:", "Let's break this down:", "Plan:", "Thought Process:", "Strategy:"
    ]
    verification_phrases = [
        "Verified with test cases.",
        "Ensured boundary cases are handled.",
        "Correctness confirmed.",
        "Double-checked logic.",
        "Validated implementation."
    ]
    
    tag_pair = random.choice(tags)
    header = random.choice(headers)
    verification = random.choice(verification_phrases)
    
    content = [
        f"{header} Developing a robust solution.",
        f"Key observation: Randomizing structure prevents collapse.",
        f"Verification: {verification}"
    ]
    random.shuffle(content)
    
    return f"<{tag_pair[0]}>\n" + "\n".join(content) + f"\n<{tag_pair[1]}>"

def generate_math_example():
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    op = random.choice(["add", "multiply", "subtract"])
    if op == "add":
        instruction = f"Write a function to add {a} and {b}."
        code = f"def solution():\n    return {a} + {b}"
    elif op == "multiply":
        instruction = f"Write a function to multiply {a} by {b}."
        code = f"def solution():\n    return {a} * {b}"
    else:
        instruction = f"Write a function to subtract {b} from {a}."
        code = f"def solution():\n    return {a} - {b}"
    return instruction, code

def generate_string_example():
    words = ["hello", "world", "python", "code", "ai", "think"]
    word = random.choice(words)
    op = random.choice(["reverse", "uppercase", "length"])
    if op == "reverse":
        instruction = f"Write a function to reverse the string '{word}'."
        code = f"def solution():\n    return '{word}'[::-1]"
    elif op == "uppercase":
        instruction = f"Write a function to uppercase the string '{word}'."
        code = f"def solution():\n    return '{word}'.upper()"
    else:
        instruction = f"Write a function to get the length of the string '{word}'."
        code = f"def solution():\n    return len('{word}')"
    return instruction, code

def generate_example():
    while True:
        gen_type = random.choice(["math", "string"])
        if gen_type == "math":
            instruction, code = generate_math_example()
        else:
            instruction, code = generate_string_example()
        
        if validate_code(code):
            break
            
    thinking = generate_thinking_block()
    response = f"{thinking}\n```python\n{code}\n```"
    
    return {
        "id": f"v5_{random.randint(100000, 999999)}",
        "instruction": instruction,
        "response": response
    }

def main():
    parser = argparse.ArgumentParser(description="Generate Dataset v5 for CodePause")
    parser.add_argument("--output", type=str, default="data/thinkanywhere_sft_v5.jsonl")
    parser.add_argument("--count", type=int, default=150)
    args = parser.parse_args()
    
    print(f"Generating {args.count} examples to {args.output}")
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        for _ in range(args.count):
            example = generate_example()
            f.write(json.dumps(example) + "\n")
            
    print(f"Successfully generated {args.count} examples.")

if __name__ == "__main__":
    main()
