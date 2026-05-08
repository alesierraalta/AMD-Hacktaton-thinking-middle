import re
import ast

def strip_thinking_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinkanywhere>.*?</thinkanywhere>", "", text, flags=re.DOTALL)
    return text

def has_balanced_tags(text: str) -> bool:
    tags = re.findall(r"<(/?)(think|thinkanywhere)>", text)
    stack = []
    for slash, tag in tags:
        if not slash:
            stack.append(tag)
        else:
            if not stack or stack[-1] != tag:
                return False
            stack.pop()
    return len(stack) == 0

def count_thinkanywhere_blocks(text: str) -> int:
    return len(re.findall(r"<thinkanywhere>.*?</thinkanywhere>", text, flags=re.DOTALL))

def extract_code(text: str) -> tuple[str, bool]:
    """
    Extract code from model output with priority:
    1. First ```python ... ``` block
    2. First ``` ... ``` block (any language)
    3. Raw output (if no blocks found)
    
    Hardened for Phase 3:
    - Strips <think> and <thinkanywhere> blocks
    - Preserves internal indentation
    - Strips leading/trailing newlines
    - Validates with ast.parse
    """
    # 1. Prefer ```python
    python_blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL)
    if python_blocks:
        code = python_blocks[0]
    else:
        # 2. Fallback to any fenced block
        any_blocks = re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.DOTALL)
        if any_blocks:
            code = any_blocks[0]
        else:
            # 3. Fallback raw output
            code = text
            
    # Strip thinking blocks from the extracted portion
    code = strip_thinking_blocks(code)
    
    # Preserve indentation but remove leading/trailing whitespace/newlines
    code = code.strip()
    
    # Validate with ast.parse
    is_valid = True
    if not code:
        is_valid = False
    else:
        try:
            import ast
            ast.parse(code)
        except SyntaxError:
            is_valid = False
            
    return code, is_valid