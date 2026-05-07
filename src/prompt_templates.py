def baseline_qwen_instruct(problem: str) -> str:
    return f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"

def thinkanywhere_qwen_instruct(problem: str) -> str:
    return f"<|im_start|>user\n{problem}\nYou may use <thinkanywhere> tags.<|im_end|>\n<|im_start|>assistant\n"

def minimal_python_function(problem: str) -> str:
    return f"Write a Python function for the following problem:\n{problem}\n\n```python\n"

def no_markdown_python_only(problem: str) -> str:
    return f"{problem}\n# Respond ONLY with valid Python code. Do not use Markdown blocks.\n"