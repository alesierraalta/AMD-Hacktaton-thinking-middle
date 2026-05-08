def baseline_qwen_instruct(problem: str) -> str:
    return f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"

def thinkanywhere_qwen_instruct(problem: str) -> str:
    return f"<|im_start|>user\n{problem}\nYou may use <thinkanywhere> tags.<|im_end|>\n<|im_start|>assistant\n"

def thinkanywhere_qwen_es(problem: str) -> str:
    """Spanish variant: instructional framing in Spanish, preserving <thinkanywhere> tags."""
    return (
        f"<|im_start|>user\n{problem}\n"
        f"Puedes usar etiquetas <thinkanywhere> para razonar.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def minimal_python_function(problem: str) -> str:
    return f"Write a Python function for the following problem:\n{problem}\n\n```python\n"

def no_markdown_python_only(problem: str) -> str:
    return f"{problem}\n# Respond ONLY with valid Python code. Do not use Markdown blocks.\n"

def get_prompt_template(template_id: str, lang: str = "en", model_family: str = "qwen"):
    """
    Select prompt template by ID with dynamic lang/model_family selection.

    Anchored on Qwen/Qwen1.5-1.8B-Chat identity caveat.
    Supports: en|es for lang, qwen for model_family.

    Template IDs:
      baseline_qwen_instruct      — bare prompt, no thinkanywhere framing
      thinkanywhere_qwen_instruct — EN thinkanywhere framing
      thinkanywhere_qwen_es       — ES thinkanywhere framing
      no_markdown_python_only    — minimal python-only instruction
      minimal_python_function    — Write a Python function preamble
    """
    if template_id in ("baseline_qwen_instruct", "baseline_qwen_chat"):
        return baseline_qwen_instruct
    if template_id in ("thinkanywhere_qwen_instruct", "thinkanywhere_qwen_chat"):
        if lang == "es":
            return thinkanywhere_qwen_es
        return thinkanywhere_qwen_instruct
    if template_id == "thinkanywhere_qwen_es":
        return thinkanywhere_qwen_es
    if template_id == "no_markdown_python_only":
        return no_markdown_python_only
    if template_id == "minimal_python_function":
        return minimal_python_function
    raise ValueError(f"Unknown template_id: {template_id!r}")


PROMPT_TEMPLATE_REGISTRY = {
    "baseline_qwen_instruct": baseline_qwen_instruct,
    "baseline_qwen_chat": baseline_qwen_instruct,
    "thinkanywhere_qwen_instruct": thinkanywhere_qwen_instruct,
    "thinkanywhere_qwen_chat": thinkanywhere_qwen_instruct,
    "thinkanywhere_qwen_es": thinkanywhere_qwen_es,
    "no_markdown_python_only": no_markdown_python_only,
    "minimal_python_function": minimal_python_function,
}