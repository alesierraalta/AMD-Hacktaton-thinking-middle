from src.prompt_templates import baseline_qwen_instruct, no_markdown_python_only

def test_baseline_qwen_instruct():
    prompt = baseline_qwen_instruct("Fix this.")
    assert "<|im_start|>user" in prompt
    assert "Fix this." in prompt
    assert "<|im_end|>" in prompt
    assert "<|im_start|>assistant" in prompt

def test_no_markdown_python_only():
    prompt = no_markdown_python_only("Fix this.")
    assert "Fix this." in prompt
    assert "Respond ONLY with valid Python code" in prompt
