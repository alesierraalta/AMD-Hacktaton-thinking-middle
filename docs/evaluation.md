# Evaluation

## Overview

Evaluation measures how well a model (baseline or fine-tuned) generates executable Python code that passes problem tests, while correctly using Think-Anywhere tags.

A single shared evaluator (`eval/evaluator.py`) powers both baseline and fine-tuned evaluation to prevent logic duplication.

## Shared Evaluator

**Entry function:** `eval.evaluator.evaluate_model(...)`

```python
def evaluate_model(
    model_name,
    problems_path,
    output_path,
    adapter_path=None,       # Phase 0.5: enables adapter-aware evaluation
    max_new_tokens=512,
    temperature=0.2,
    mock=False,              # Local verification without GPU
    timeout=5,
):
```

**Responsibilities:**
1. Load problems from JSONL.
2. Load model + tokenizer (or skip in mock mode).
3. Generate (or mock) raw output for each problem.
4. Strip thinking blocks.
5. Execute in sandbox against problem tests.
6. Compute metrics.
7. Write results JSONL and return summary.

## CLI Entry Points

| Script | Purpose | Typical Args |
|--------|---------|--------------|
| `eval/evaluate_baseline.py` | Evaluate base model | `--model_name`, `--problems_path`, `--output_path`, `--mock` |
| `eval/evaluate_finetuned.py` | Evaluate base + LoRA adapter | `--base_model`, `--adapter_path`, `--problems_path`, `--output_path`, `--mock` |
| `eval/make_report.py` | Generate markdown report | `--input_path` (single) or `--baseline` + `--finetuned` + `--out` |

## Mock Mode

Mock mode replaces model inference with deterministic tagged output. It exists so the full pipeline can be verified locally before GPU time is spent.

**Enable:** Add `--mock` to either evaluation script.

**Mock output format:**
```text
<think>Plan: implement {entry_point} carefully.</think>
```python
def {entry_point}():
    <thinkanywhere>Validate edge cases here.</thinkanywhere>
    pass
```
```

**Mock verification commands:**

```bash
python eval/evaluate_baseline.py --mock --problems_path data/problems.jsonl --output_path results/baseline_mock.jsonl
python eval/evaluate_finetuned.py --mock --base_model mock-base --adapter_path mock-adapter --problems_path data/problems.jsonl --output_path results/finetuned_mock.jsonl
```

## JSONL Output Schema

Each line in the output JSONL:

```json
{
  "id": 0,
  "prompt": "Write a function add(a, b)...",
  "raw_output": "<think>...</think>\ndef add(a, b):...",
  "clean_code": "def add(a, b):...",
  "passed": true,
  "metrics": {
    "tests_passed": true,
    "balanced_tags": true,
    "has_thinkanywhere": true,
    "thinkanywhere_blocks": 1,
    "executable_after_strip": true,
    "clean_code_lines": 3
  }
}
```

| Field | Description |
|-------|-------------|
| `id` | Problem identifier |
| `prompt` | Original problem text |
| `raw_output` | Model output before stripping |
| `clean_code` | Code after stripping thinking blocks |
| `passed` | Whether all tests passed |
| `metrics` | Six computed metrics (see `docs/architecture.md`) |

## Report Generation

### Single Report

```bash
python eval/make_report.py --input_path results/baseline.jsonl --out results/report.md
```

Contains: total problems, passed count, pass rate, per-problem PASS/FAIL list.

### Comparison Report

```bash
python eval/make_report.py \
  --baseline results/baseline.jsonl \
  --finetuned results/finetuned.jsonl \
  --out results/comparison.md
```

Contains: side-by-side table of baseline vs. fine-tuned totals, passed, and pass rates; per-problem PASS/FAIL comparison.

## Evaluation Checklist

Before GPU execution:
- [ ] `evaluate_baseline.py --mock` produces 30-record JSONL.
- [ ] `evaluate_finetuned.py --mock` produces 30-record JSONL.
- [ ] `make_report.py` generates valid markdown from both.
- [ ] All `metrics` fields are present in every record.
- [ ] `passed` aligns with `metrics.tests_passed`.

## GPU Evaluation Commands

See [`sdd/codepause-phase-1-gpu-runbook.md`](../sdd/codepause-phase-1-gpu-runbook.md) for the authoritative GPU execution steps.

Quick reference:

```bash
# Baseline
python eval/evaluate_baseline.py \
  --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --problems_path data/problems.jsonl \
  --output_path results/baseline_gpu.jsonl

# Fine-tuned
python eval/evaluate_finetuned.py \
  --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --adapter_path outputs/qwen25-coder-codepause-lora \
  --problems_path data/problems.jsonl \
  --output_path results/finetuned_gpu.jsonl

# Report
python eval/make_report.py \
  --baseline results/baseline_gpu.jsonl \
  --finetuned results/finetuned_gpu.jsonl \
  --out results/phase1_report.md
```
