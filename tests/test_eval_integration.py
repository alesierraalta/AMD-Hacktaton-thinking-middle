import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.evaluator import evaluate_model, _get_prompt, PROMPT_TEMPLATES

def test_prompt_template_selection():
    problem = {"prompt": "Hello world"}
    
    # Test default
    prompt = _get_prompt(problem)
    assert "### Instruction\nHello world" in prompt
    
    # Test baseline_qwen_instruct
    prompt = _get_prompt(problem, "baseline_qwen_instruct")
    assert "<|im_start|>user\nHello world" in prompt
    
    # Test thinkanywhere_qwen_instruct
    prompt = _get_prompt(problem, "thinkanywhere_qwen_instruct")
    assert "You may use <thinkanywhere> tags." in prompt
    
    # Test no_markdown_python_only
    prompt = _get_prompt(problem, "no_markdown_python_only")
    assert "# Respond ONLY with valid Python code." in prompt

def test_evaluator_mock_sanity(tmp_path):
    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(json.dumps({
        "id": "test_1",
        "prompt": "Add two numbers",
        "entry_point": "add",
        "tests": ["assert add(1, 1) == 2"]
    }) + "\n")
    
    output_file = tmp_path / "results.jsonl"
    
    # Mocking a success case where mock generate produces valid code for the test
    # Note: evaluate_model uses _mock_generate which produces a generic 'pass' function.
    # To actually 'pass' the test, I'd need to mock _mock_generate or run_code.
    # But here I want to test the wiring and prompt template passing.
    
    summary = evaluate_model(
        model_name="mock",
        problems_path=str(problems_file),
        output_path=str(output_file),
        mock=True,
        prompt_template="baseline_qwen_instruct"
    )
    
    assert summary["total"] == 1
    # It will fail the test because _mock_generate returns 'pass'
    assert summary["passed"] == 0
    
    with open(output_file, "r") as f:
        record = json.loads(f.readline())
        assert record["id"] == "test_1"
        assert "<|im_start|>user" in record["raw_output"]
        assert "def add():" in record["clean_code"]

def test_evaluator_ast_bypass(tmp_path):
    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(json.dumps({
        "id": "test_syntax_error",
        "prompt": "Failing syntax",
        "entry_point": "fail",
        "tests": ["assert True"]
    }) + "\n")
    
    output_file = tmp_path / "results.jsonl"
    
    # We need to mock _mock_generate to return invalid syntax
    import eval.evaluator
    original_mock = eval.evaluator._mock_generate
    eval.evaluator._mock_generate = lambda p, t: "INVALID PYTHON CODE (no function def)"
    
    try:
        summary = evaluate_model(
            model_name="mock",
            problems_path=str(problems_file),
            output_path=str(output_file),
            mock=True
        )
        
        assert summary["passed"] == 0
        with open(output_file, "r") as f:
            record = json.loads(f.readline())
            assert record["metrics"]["executable_after_strip"] == False
            # Check for our explicit error message injected in evaluate_model
            # Note: evaluate_model in my edit set test_result = {"passed": False, "error": "AST SyntaxError"}
            # And metrics uses test_result.
    finally:
        eval.evaluator._mock_generate = original_mock
