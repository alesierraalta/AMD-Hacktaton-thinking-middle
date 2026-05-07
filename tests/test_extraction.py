from src.strip_thinking import extract_code

def test_extract_python_block():
    text = "Here is my code:\n```python\ndef foo():\n    return 1\n```\nHope it helps!"
    code, is_valid = extract_code(text)
    assert code == "def foo():\n    return 1"
    assert is_valid is True

def test_extract_any_block():
    text = "Here is my code:\n```\ndef foo():\n    return 1\n```\nHope it helps!"
    code, is_valid = extract_code(text)
    assert code == "def foo():\n    return 1"
    assert is_valid is True

def test_extract_raw():
    text = "def foo():\n    return 1"
    code, is_valid = extract_code(text)
    assert code == "def foo():\n    return 1"
    assert is_valid is True

def test_strip_thinking_from_extracted():
    text = "```python\ndef foo():\n    <think>thinking</think>\n    <thinkanywhere>also thinking</thinkanywhere>\n    return 1\n```"
    code, is_valid = extract_code(text)
    assert "<think" not in code
    assert is_valid is True

def test_syntax_error():
    text = "```python\ndef foo() return 1\n```"
    code, is_valid = extract_code(text)
    assert is_valid is False
