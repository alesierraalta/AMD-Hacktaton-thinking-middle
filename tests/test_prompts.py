import pytest
from src.prompts import build_sft_prompt, build_baseline_prompt


class TestBuildSftPrompt:
    def test_contains_instruction_and_response(self):
        problem = {
            "id": "test_001",
            "prompt": "Write a function add(a, b).",
            "entry_point": "add",
            "tests": ["assert add(1, 2) == 3"],
        }
        result = build_sft_prompt(problem)
        assert "### Instruction" in result
        assert "### Response" in result
        assert "add(a, b)" in result

    def test_contains_think_and_thinkanywhere(self):
        problem = {
            "id": "test_002",
            "prompt": "Write a function.",
            "entry_point": "foo",
            "tests": [],
        }
        result = build_sft_prompt(problem)
        assert "<think>" in result
        assert "</think>" in result
        assert "<thinkanywhere>" in result
        assert "</thinkanywhere>" in result


class TestBuildBaselinePrompt:
    def test_contains_problem_prompt(self):
        problem = {
            "id": "test_003",
            "prompt": "Write a function sub(a, b).",
            "entry_point": "sub",
            "tests": ["assert sub(5, 2) == 3"],
        }
        result = build_baseline_prompt(problem)
        assert "sub(a, b)" in result
        assert "### Instruction" in result

    def test_no_response_section(self):
        problem = {
            "id": "test_004",
            "prompt": "Write a function.",
            "entry_point": "foo",
            "tests": [],
        }
        result = build_baseline_prompt(problem)
        assert "### Response" not in result
