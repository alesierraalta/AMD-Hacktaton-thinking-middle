import pytest
import json
import os
from src.strip_thinking import extract_code
from src.prompt_templates import get_prompt_template

def test_extract_code_priority_python():
    text = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```"
    code, valid = extract_code(text)
    assert code == "def add(a, b):\n    return a + b"
    assert valid is True

def test_extract_code_priority_any_block():
    text = "Some text\n```\ndef sub(a, b):\n    return a - b\n```"
    code, valid = extract_code(text)
    assert code == "def sub(a, b):\n    return a - b"
    assert valid is True

def test_extract_code_raw_fallback():
    text = "def mul(a, b):\n    return a * b"
    code, valid = extract_code(text)
    assert code == "def mul(a, b):\n    return a * b"
    assert valid is True

def test_extract_code_strips_thinking():
    text = "<think>reasoning</think>```python\ndef foo():\n    <thinkanywhere>more reasoning</thinkanywhere>\n    return 1\n```"
    code, valid = extract_code(text)
    assert "think" not in code
    assert "reasoning" not in code
    assert code == "def foo():\n    \n    return 1"
    assert valid is True

def test_extract_code_invalid_syntax():
    text = "```python\ndef unfinished("
    code, valid = extract_code(text)
    assert valid is False

def test_prompt_template_minimal_python_function():
    tmpl = get_prompt_template("minimal_python_function")
    result = tmpl("Add two numbers.")
    assert "Write a Python function" in result
    assert "Add two numbers." in result
    assert "```python" in result

def test_prompt_template_no_markdown_python_only():
    tmpl = get_prompt_template("no_markdown_python_only")
    result = tmpl("Add two numbers.")
    assert "Add two numbers." in result
    assert "ONLY with valid Python code" in result
    assert "Markdown" in result
