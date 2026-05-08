import argparse
import os
import json
import torch
from eval.evaluator import run_ablation, evaluate_model, _build_metadata

def run_phase3(mock=False):
    print(f"--- Phase 3: Generalization & Ablation (mock={mock}) ---")
    
    # 1. Hardware Check
    device_name = "None"
    if not mock:
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. Hardware check FAILED.")
            return
        device_name = torch.cuda.get_device_name(0)
        if "Tesla T4" not in device_name:
            print(f"WARNING: Device is {device_name}, expected Tesla T4. Proceeding with caution.")
        else:
            print(f"Hardware Check PASSED: {device_name}")
    else:
        print("Hardware Check SKIPPED (MOCK MODE)")
        device_name = "MOCK-T4"

    # 2. Artifacts Check
    adapter_path = "drive/codepause_phase2_v2/phase2_recipe_a"
    base_model_name = "Qwen/Qwen1.5-1.8B-Chat"
    heldout_path = "data/heldout_problems_30.jsonl"
    
    if not mock:
        if not os.path.exists(adapter_path):
            print(f"ERROR: Adapter missing at {adapter_path}")
            return
    if not os.path.exists(heldout_path):
        print(f"ERROR: Held-out dataset missing at {heldout_path}")
        return
    print("Artifacts Check PASSED")

    # 3. Main evaluation on held-out dataset
    print("\nRunning Main Held-out Evaluation...")
    
    # Baseline
    print("--- Held-out Baseline ---")
    baseline_metadata = _build_metadata(base_model_name, None, "baseline_qwen_instruct", "en", hardware=device_name)
    baseline_summary = evaluate_model(
        model_name=base_model_name,
        problems_path=heldout_path,
        output_path="results/phase3_heldout_baseline.jsonl",
        metadata=baseline_metadata,
        prompt_template="baseline_qwen_instruct",
        mock=mock
    )
    
    # Finetuned
    print("--- Held-out Finetuned ---")
    finetuned_metadata = _build_metadata(base_model_name, adapter_path, "thinkanywhere_qwen_instruct", "en", hardware=device_name)
    finetuned_summary = evaluate_model(
        model_name=base_model_name,
        problems_path=heldout_path,
        output_path="results/phase3_heldout_finetuned.jsonl",
        adapter_path=adapter_path,
        metadata=finetuned_metadata,
        prompt_template="thinkanywhere_qwen_instruct",
        mock=mock
    )

    # 4. Ablation Study (4 arms A/B/C/D)
    print("\nRunning 4-arm Ablation Study...")
    ablation_results = run_ablation(
        problems_path=heldout_path,
        output_dir="results/ablation",
        base_model_name=base_model_name,
        adapter_path=adapter_path,
        mock=mock
    )

    # 5. Final Report Generation
    print("\nGenerating Phase 3 Final Report...")
    report = f"""# Phase 3 Generalization & Ablation Report

## Executive Summary
This report presents results for model generalization on 30 unseen (held-out) problems and a systematic ablation study of prompting vs. adapter influences.

## Held-out Results (Generalization)
- **Baseline**: {baseline_summary['passed']}/{baseline_summary['total']} ({baseline_summary['pass_rate']:.1f}%)
- **Fine-tuned**: {finetuned_summary['passed']}/{finetuned_summary['total']} ({finetuned_summary['pass_rate']:.1f}%)
- **Delta**: {finetuned_summary['passed'] - baseline_summary['passed']} problems ({finetuned_summary['pass_rate'] - baseline_summary['pass_rate']:.1f} pp)

## Ablation Results (4 Arms)
- **Arm A (Base + Baseline Prompt)**: {ablation_results['A']['pass_rate']:.1f}%
- **Arm B (Base + ThinkAnywhere Prompt)**: {ablation_results['B']['pass_rate']:.1f}%
- **Arm C (Adapter + ThinkAnywhere Prompt)**: {ablation_results['C']['pass_rate']:.1f}%
- **Arm D (Adapter + Minimal Python)**: {ablation_results['D']['pass_rate']:.1f}%

## Hardware & Environment
- Model: {base_model_name}
- Hardware: {device_name}
- Mock: {mock}
- Notebook: codepause_phase_3_generalization_ablation.ipynb

## Notes
- Phase 2 v2 results for comparison: 24/30 -> 25/30 (+3.3pp).
- No retraining was performed in this phase.
"""
    with open("results/phase3_final_report.md", "w") as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    run_phase3(mock=args.mock)

