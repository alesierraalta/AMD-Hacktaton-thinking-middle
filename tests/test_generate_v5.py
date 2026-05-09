import pytest
from scripts.generate_sft_v5 import validate_code

def test_validate_code_accepts_valid_code():
    code = "def add(a, b):\n    return a + b"
    assert validate_code(code) is True

def test_validate_code_rejects_pass():
    code = "def add(a, b):\n    pass"
    assert validate_code(code) is False

def test_validate_code_rejects_placeholder_comment():
    code = "def add(a, b):\n    # implementation here\n    return 0"
    assert validate_code(code) is False

def test_validate_code_rejects_todo():
    code = "def add(a, b):\n    # TODO: implement\n    return 0"
    assert validate_code(code) is False

def test_validate_code_case_insensitive():
    code = "def add(a, b):\n    # implementation\n    return 0"
    assert validate_code(code) is False

def test_generate_thinking_block_contains_tags():
    from scripts.generate_sft_v5 import generate_thinking_block
    thinking = generate_thinking_block()
    # Should start with an opening tag and end with a closing tag
    assert thinking.startswith("<")
    assert thinking.endswith(">")
    assert "\n" in thinking

def test_generate_thinking_block_is_diverse():
    from scripts.generate_sft_v5 import generate_thinking_block
    blocks = [generate_thinking_block() for _ in range(10)]
    # At least some should be different
    assert len(set(blocks)) > 1

def test_generate_example_returns_dict_with_keys():
    from scripts.generate_sft_v5 import generate_example
    example = generate_example()
    assert "instruction" in example
    assert "response" in example
    assert "id" in example

def test_generate_example_code_is_valid():
    from scripts.generate_sft_v5 import generate_example, validate_code
    example = generate_example()
    # Extract code from response
    import re
    match = re.search(r"```python\n(.*?)```", example["response"], re.DOTALL)
    assert match is not None
    code = match.group(1)
    assert validate_code(code) is True

def test_generate_sft_v5_cli_produces_file(tmp_path):
    import subprocess
    import os
    output_file = tmp_path / "test_v5.jsonl"
    result = subprocess.run(
        ["python", "scripts/generate_sft_v5.py", "--output", str(output_file), "--count", "5"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."}
    )
    assert result.returncode == 0
    assert output_file.exists()
    # Check number of lines
    with open(output_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 5
    # Check if first line is valid JSON
    import json
    data = json.loads(lines[0])
    assert "instruction" in data
    assert "response" in data
