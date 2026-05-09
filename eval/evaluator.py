import argparse
import json
import os
import gc

from src.model_loader import load_model_and_tokenizer
from src.prompts import build_baseline_prompt
from src.strip_thinking import extract_code
from src.sandbox_runner import run_code
from src.metrics import compute_metrics
from src.prompt_templates import (
    baseline_qwen_instruct,
    thinkanywhere_qwen_instruct,
    thinkanywhere_qwen_es,
    no_markdown_python_only,
    minimal_python_function,
    get_prompt_template,
)

PROMPT_TEMPLATES = {
    "baseline_qwen_instruct": baseline_qwen_instruct,
    "thinkanywhere_qwen_instruct": thinkanywhere_qwen_instruct,
    "thinkanywhere_qwen_es": thinkanywhere_qwen_es,
    "no_markdown_python_only": no_markdown_python_only,
    "minimal_python_function": minimal_python_function,
}

# ----------------------------------------------------------------------
# Ablation arm definitions
# ----------------------------------------------------------------------
ABLATION_ARMS = [
    {
        "arm": "A",
        "label": "base_model_plus_base_prompt",
        "model_family": "qwen",
        "prompt_template_id": "baseline_qwen_instruct",
        "language": "en",
    },
    {
        "arm": "B",
        "label": "base_model_plus_thinkanywhere_prompt",
        "model_family": "qwen",
        "prompt_template_id": "thinkanywhere_qwen_instruct",
        "language": "en",
    },
    {
        "arm": "C",
        "label": "finetuned_adapter_plus_thinkanywhere_prompt",
        "model_family": "qwen",
        "prompt_template_id": "thinkanywhere_qwen_instruct",
        "language": "en",
    },
    {
        "arm": "D",
        "label": "finetuned_adapter_plus_minimal_python",
        "model_family": "qwen",
        "prompt_template_id": "minimal_python_function",
        "language": "en",
    },
]


def _build_metadata(model_name, adapter_path, prompt_template_id, language, hardware="Colab T4"):
    return {
        "model_id": model_name,
        "adapter_path": adapter_path or "none",
        "prompt_template_id": prompt_template_id,
        "language": language,
        "hardware": hardware,
    }


def _clear_model_cache():
    """Clear model cache to prevent T4 OOM between ablation arms."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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


def run_ablation(
    problems_path: str,
    output_dir: str,
    base_model_name: str,
    adapter_path: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    mock: bool = False,
    timeout: int = 5,
):
    """
    Run the 4-arm ablation matrix sequentially with cache clearing between arms.
    """
    # Phase 3 Hard-stop: Validate dataset before starting
    print(f"Validating dataset: {problems_path}")
    from eval.validate_dataset import validate_dataset
    try:
        validate_dataset(problems_path, schema="problems")
    except SystemExit:
        print("ERROR: Dataset validation failed. Stopping ablation.")
        raise RuntimeError("Dataset validation failed.")

    # Phase 3 Hard-stop: Adapter Probe
    if adapter_path and not mock:
        print(f"Running adapter probe hard-stop for: {adapter_path}")
        from eval.adapter_probe import probe_adapter
        probe_prompt = "Write a python function to add two numbers. You may use <thinkanywhere> tags."
        if not probe_adapter(base_model_name, adapter_path, probe_prompt):
            print("ERROR: Adapter probe failed (no difference detected). Hard-stop triggered.")
            raise RuntimeError("Adapter probe failed: Model output did not change with adapter.")
        print("Adapter probe PASSED.")
    
    os.makedirs(output_dir, exist_ok=True)
    arm_results = {}

    for arm_def in ABLATION_ARMS:
        arm = arm_def["arm"]
        label = arm_def["label"]
        prompt_template_id = arm_def["prompt_template_id"]
        language = arm_def["language"]

        print(f"\n{'='*60}")
        print(f"ARM {arm}: {label}")
        print(f"{'='*60}")

        arm_adapter = adapter_path if arm in ("C", "D") else None
        arm_model = base_model_name

        metadata = _build_metadata(
            model_name=arm_model,
            adapter_path=arm_adapter,
            prompt_template_id=prompt_template_id,
            language=language,
        )

        output_path = os.path.join(output_dir, f"arm_{arm}.jsonl")
        summary = evaluate_model(
            model_name=arm_model,
            problems_path=problems_path,
            output_path=output_path,
            adapter_path=arm_adapter,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            mock=mock,
            timeout=timeout,
            metadata=metadata,
            prompt_template=prompt_template_id,
        )

        print(f"ARM {arm} complete — {summary['passed']}/{summary['total']} passed "
              f"({summary['pass_rate']:.1f}%). Loops: {summary['thinking_loops']}, Lazy: {summary['lazy_outputs']}")

        arm_results[arm] = summary

        if not mock:
            _clear_model_cache()

    return arm_results


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
    loops_count = 0
    lazy_count = 0

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
            raw_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

        clean_code, is_valid = extract_code(raw_output)
        if not is_valid:
            test_result = {"passed": False, "error": "AST SyntaxError"}
        else:
            tests = problem.get("tests", [])
            test_result = run_code(clean_code, tests, timeout=timeout)
        
        metrics = compute_metrics(clean_code, raw_output, test_result)
        
        if metrics.get("has_thinking_loop"):
            loops_count += 1
        if metrics.get("is_lazy"):
            lazy_count += 1

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
        "thinking_loops": loops_count,
        "lazy_outputs": lazy_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model or run ablation")
    parser.add_argument("--problems_path", type=str, default="data/problems.jsonl")
    parser.add_argument("--output_path", type=str, default="results/eval.jsonl")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-1.8B-Chat")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--ablation", action="store_true",
                        help="Run 4-arm ablation instead of single evaluation")
    parser.add_argument("--output_dir", type=str, default="results/ablation",
                        help="Directory for ablation output JSONL files")
    args = parser.parse_args()

    if args.ablation:
        run_ablation(
            problems_path=args.problems_path,
            output_dir=args.output_dir,
            base_model_name=args.model_name,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            mock=args.mock,
            timeout=args.timeout,
        )
    else:
        metadata = _build_metadata(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            prompt_template_id="baseline_qwen_instruct",
            language="en",
        )
        summary = evaluate_model(
            model_name=args.model_name,
            problems_path=args.problems_path,
            output_path=args.output_path,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            mock=args.mock,
            timeout=args.timeout,
            metadata=metadata,
        )
        print(f"Results: {summary['passed']}/{summary['total']} passed "
              f"({summary['pass_rate']:.1f}%). Loops: {summary['thinking_loops']}, Lazy: {summary['lazy_outputs']}")


if __name__ == "__main__":
    main()
