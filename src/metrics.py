import ast
import re
from src.strip_thinking import has_balanced_tags, count_thinkanywhere_blocks

# Thresholds for loop detection
MAX_LINE_REPETITIONS = 4  # Increased from 3
MIN_SEQUENCE_LENGTH = 3   # Increased from 2
MIN_NGRAM_WORDS = 5       # Increased from 3

def has_thinking_loops(raw_output: str) -> bool:
    """Detects repetitive text loops within the thinking blocks."""
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if not lines:
        return False
    
    # Check for line-level repetition
    line_counts = {}
    for line in lines:
        if len(line) < 10: continue # Ignore very short lines like "Plan:"
        line_counts[line] = line_counts.get(line, 0) + 1
        if line_counts[line] >= MAX_LINE_REPETITIONS:
            return True
            
    # Check for sequence repetition (sliding window of lines)
    for i in range(len(lines) - (MIN_SEQUENCE_LENGTH * 2) + 1):
        seq = tuple(lines[i:i+MIN_SEQUENCE_LENGTH])
        if len(" ".join(seq)) < 20: continue # Ignore short sequences
        for j in range(i + MIN_SEQUENCE_LENGTH, len(lines) - MIN_SEQUENCE_LENGTH + 1):
            if tuple(lines[j:j+MIN_SEQUENCE_LENGTH]) == seq:
                return True
                
    # Basic string-level n-gram check
    words = raw_output.split()
    for i in range(len(words) - (MIN_NGRAM_WORDS * 2)):
        phrase = " ".join(words[i:i+MIN_NGRAM_WORDS])
        if len(phrase) < 25: continue # Ignore short phrases
        # Check if phrase repeats in the next 20 words
        if phrase in " ".join(words[i+MIN_NGRAM_WORDS : i+MIN_NGRAM_WORDS+20]):
            return True

    return False


def is_lazy_output(clean_code: str, problem_type: str = None) -> bool:
    """
    Detects if the generated code is a hardcoded or trivial response.
    """
    clean_code = clean_code.strip()
    if not clean_code:
        return True
    
    # Check for simple literals
    if clean_code.lower() in ("true", "false", "none", "true.", "false.", "\"true\"", "'true'"):
        return True
    
    try:
        tree = ast.parse(clean_code)
        if len(tree.body) == 1:
            stmt = tree.body[0]
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant):
                return True
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                return True
    except:
        if len(clean_code) < 15 and any(x in clean_code for x in ["True", "False"]):
            return True
            
    return False


def compute_metrics(clean_code: str, raw_output: str, test_result: dict) -> dict:
    balanced = has_balanced_tags(raw_output)
    thinkanywhere_count = count_thinkanywhere_blocks(raw_output)
    
    thinking_loop = has_thinking_loops(raw_output)
    lazy = is_lazy_output(clean_code)

    try:
        compile(clean_code, "<string>", "exec")
        executable = len(clean_code.strip()) > 0
    except SyntaxError:
        executable = False

    return {
        "tests_passed": test_result.get("passed", False),
        "balanced_tags": balanced,
        "has_thinkanywhere": thinkanywhere_count > 0,
        "thinkanywhere_blocks": thinkanywhere_count,
        "executable_after_strip": executable,
        "clean_code_lines": len(clean_code.splitlines()),
        "has_thinking_loop": thinking_loop,
        "is_lazy": lazy,
    }