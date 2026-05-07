import json
import os

from src.model_loader import load_model_and_tokenizer
from src.prompts import build_baseline_prompt
from src.strip_thinking import extract_code
from src.sandbox_runner import run_code
from src.metrics import compute_metrics
from src.prompt_templates import (
    baseline_qwen_instruct,
    thinkanywhere_qwen_instruct,
    no_markdown_python_only,
)

PROMPT_TEMPLATES = {
    "baseline_qwen_instruct": baseline_qwen_instruct,
    "thinkanywhere_qwen_instruct": thinkanywhere_qwen_instruct,
    "no_markdown_python_only": no_markdown_python_only,
}


def _load_problems(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Problems file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _get_prompt(problem: dict, template_name: str = None) -> str:
    instruction = problem.get("prompt", "")
    if template_name and template_name in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[template_name](instruction)
    return build_baseline_prompt(problem)


def _mock_generate(problem: dict, template_name: str = None) -> str:
    prompt = _get_prompt(problem, template_name)
    entry_point = problem.get("entry_point", "solve")
    raw = (
        f"{prompt}\n"
        f"<think>Plan: implement {entry_point} carefully.</think>\n"
        f"```python\n"
        f"def {entry_point}():\n"
        f"    <thinkanywhere>Validate edge cases here.</thinkanywhere>\n"
        f"    pass\n"
        f"```"
    )
    return raw


def evaluate_model(
    model_name,
    problems_path,
    output_path,
    adapter_path=None,
    max_new_tokens=512,
    temperature=0.2,
    mock=False,
    timeout=5,
    metadata=None,
    prompt_template=None,
):
    """
    Evaluate a model (or mock) on problems and write JSONL results.

    metadata: optional dict injected into every result record under an
              optional "metadata" key. Used by make_report.py to surface
              pipeline provenance (model_name, adapter_path, hardware, etc.)
              without changing the existing schema fields.
    """
    problems = _load_problems(problems_path)

    model = None
    tokenizer = None
    if not mock:
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            adapter_path=adapter_path,
            device_map="auto",
            torch_dtype="auto",
        )

    results = []
    total_problems = len(problems)
    for index, problem in enumerate(problems, start=1):
        problem_id = problem.get("id", index)
        print(f"Evaluating problem {index}/{total_problems}: {problem_id}", flush=True)
        if mock:
            raw_output = _mock_generate(problem, prompt_template)
        else:
            prompt = _get_prompt(problem, prompt_template)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        clean_code, is_valid = extract_code(raw_output)
        if not is_valid:
            test_result = {"passed": False, "error": "AST SyntaxError"}
        else:
            tests = problem.get("tests", [])
            test_result = run_code(clean_code, tests, timeout=timeout)
        
        metrics = compute_metrics(clean_code, raw_output, test_result)
        record = {
            "id": problem.get("id", ""),
            "prompt": problem.get("prompt", ""),
            "raw_output": raw_output,
            "clean_code": clean_code,
            "passed": test_result.get("passed", False),
            "metrics": metrics,
        }
        if metadata:
            record["metadata"] = metadata
        results.append(record)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    return {
        "total": total,
        "passed": passed,
        "pass_rate": (passed / total * 100.0) if total else 0.0,
    }
