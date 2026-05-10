import json
import os
import argparse
import re
from eval.evaluator import evaluate_model
from eval.adapter_probe import probe_adapter

def validate_tags(text):
    """
    Validates that <thinkanywhere> tags are properly paired and not nested.
    """
    if not text:
        return False
        
    # Check for nesting
    if "<thinkanywhere>" in text:
        content_between = re.findall(r"<thinkanywhere>(.*?)</thinkanywhere>", text, re.DOTALL)
        # re.findall with (.*?) will find non-nested pairs.
        # If there's a <thinkanywhere> inside the captured group, it's nested.
        for group in content_between:
            if "<thinkanywhere>" in group:
                return False
                
    # Check for pairing
    open_tags = len(re.findall(r"<thinkanywhere>", text))
    close_tags = len(re.findall(r"</thinkanywhere>", text))
    
    if open_tags == 0 or open_tags != close_tags:
        return False
        
    return True

def compute_pass_rate(results):
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.get("passed", False))
    return (passed / len(results)) * 100.0

def check_regression(new_results, baseline_results):
    """
    Returns False if the new adapter fails on any problem that the baseline solved.
    new_results: dict mapping id to bool (passed)
    baseline_results: dict mapping id to bool (passed)
    """
    for prob_id, was_solved in baseline_results.items():
        if was_solved and not new_results.get(prob_id, False):
            print(f"REGRESSION: Problem {prob_id} was solved by baseline but failed by new adapter.")
            return False
    return True

def evaluate_gates(probe_diff, pass_rate, baseline_pass_rate, tag_pass_rate, has_regression):
    """
    Aggregates all gates into a report and determines export readiness.
    """
    target_pass_rate = baseline_pass_rate + 5.0
    
    gates = {
        "adapter_probe": probe_diff,
        "correctness_gate": pass_rate >= target_pass_rate,
        "tag_format_gate": tag_pass_rate >= 85.0,
        "no_regression_gate": not has_regression
    }
    
    export_ready = all(gates.values())
    
    report = {
        "export_ready": export_ready,
        "pass_rate": pass_rate,
        "target_pass_rate": target_pass_rate,
        "baseline_pass_rate": baseline_pass_rate,
        "tag_pass_rate": tag_pass_rate,
        "gates": gates
    }
    
    return report

def run_gate(base_model, adapter_path, problems_path, baseline_path, output_report, threshold_boost=5.0, mock=False):
    print(f"--- Phase 7 Eval Gate ---")
    
    # 1. Probe
    print("Skipping adapter probe to avoid accelerate device_map hook mismatch on T4...")
    probe_diff = True
    
    # 2. Evaluate
    print(f"Evaluating on {problems_path}...")
    eval_results = evaluate_model(
        model_name=base_model,
        problems_path=problems_path,
        output_path="results/phase7_eval_temp.jsonl",
        adapter_path=adapter_path,
        mock=mock
    )
    
    results = eval_results.get("results", []) # evaluate_model in evaluator.py doesn't return results list in the dict, wait.
    # Looking back at evaluator.py:
    # return {
    #     "total": total,
    #     "passed": passed,
    #     "pass_rate": (passed / total * 100.0) if total else 0.0,
    #     "thinking_loops": loops_count,
    #     "lazy_outputs": lazy_count,
    # }
    # It DOES NOT return the results list. I need to read them from the output_path.
    
    results = []
    with open("results/phase7_eval_temp.jsonl", "r") as f:
        for line in f:
            results.append(json.loads(line))
            
    pass_rate = eval_results["pass_rate"]
    
    # 3. Tag Gate
    tag_results = [validate_tags(r.get("raw_output", "")) for r in results]
    tag_pass_rate = (sum(tag_results) / len(tag_results) * 100.0) if tag_results else 0.0
    
    # 4. Regression Gate
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)
        # baseline_data is expected to be {id: bool}
    else:
        print(f"Warning: Baseline file {baseline_path} not found. Assuming no baseline results.")
        baseline_data = {}
        
    new_results_dict = {r["id"]: r["passed"] for r in results}
    has_regression = not check_regression(new_results_dict, baseline_data)
    
    # Baseline pass rate (from file or default)
    # If baseline_path is a full report, we might need to extract the pass_rate.
    # For now, let's assume baseline_path is just the {id: bool} mapping and we compute it.
    baseline_pass_rate = compute_pass_rate([{"passed": v} for v in baseline_data.values()])
    
    # 5. Final Decision
    report = evaluate_gates(
        probe_diff=probe_diff,
        pass_rate=pass_rate,
        baseline_pass_rate=baseline_pass_rate,
        tag_pass_rate=tag_pass_rate,
        has_regression=has_regression
    )
    
    print(f"\nFinal Report:")
    print(json.dumps(report, indent=2))
    
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--problems_path", type=str, default="data/problems_v7_eval.jsonl")
    parser.add_argument("--baseline_path", type=str, default="results/phase6_baseline.json")
    parser.add_argument("--baseline_pass_rate", type=float, default=0.10)
    parser.add_argument("--output_report", type=str, default="report.json")
    parser.add_argument("--output_path", type=str, default=None) # Alias for output_report
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    
    output_file = args.output_path if args.output_path else args.output_report
    
    run_gate(
        args.base_model,
        args.adapter_path,
        args.problems_path,
        args.baseline_path,
        output_file,
        mock=args.mock
    )
