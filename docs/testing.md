# Testing

## Test Commands

```bash
# Full suite with coverage
python -m pytest --cov=src --cov=training --cov=eval --cov=app --cov-report=term-missing
```

Expected: **95 passed**, **91% total coverage**.

```bash
# Fast run without coverage
python -m pytest
```

```bash
# Specific module
python -m pytest tests/test_evaluator.py -v
```

## Test Inventory

| File | Count | What It Protects |
|------|-------|------------------|
| `tests/test_strip_thinking.py` | 20 | Tag stripping, balance detection, block counting |
| `tests/test_sandbox_runner.py` | 5 | Subprocess execution, timeout handling, temp file cleanup |
| `tests/test_metrics.py` | 6 | Metric computation, syntax validation, edge cases |
| `tests/test_prompts.py` | 4 | Prompt builder output format |
| `tests/test_model_loader.py` | 5 | Base model loading, adapter application, missing-PEFT error |
| `tests/test_evaluator.py` | 10 | Shared evaluator mock mode, real mode, adapter path forwarding, timeout propagation |
| `tests/test_eval_baseline.py` | 5 | Baseline CLI argument parsing and execution |
| `tests/test_eval_finetuned.py` | 6 | Fine-tuned CLI argument parsing and execution |
| `tests/test_eval_report.py` | 8 | Single report, comparison report, missing file handling |
| `tests/test_pipeline.py` | 9 | End-to-end flow, dataset validation, SFT tag balance, strip-to-executable, pass-rate gate |
| `tests/test_training_sft_lora.py` | 7 | CLI args, dry-run, missing dataset, help output |
| `tests/test_training_rewards.py` | 4 | Reward stub import and shape |
| `tests/test_demo_mock.py` | 6 | Demo pipeline, Gradio fallback, metric output |

**Total: 95 tests**

## Coverage by Module

| Module | Stmts | Miss | Cover |
|--------|-------|------|-------|
| `app/demo_mock.py` | 32 | 6 | 81% |
| `eval/evaluate_baseline.py` | 21 | 1 | 95% |
| `eval/evaluate_finetuned.py` | 22 | 1 | 95% |
| `eval/evaluator.py` | 43 | 0 | 100% |
| `eval/make_report.py` | 60 | 2 | 97% |
| `src/metrics.py` | 10 | 0 | 100% |
| `src/model_loader.py` | 21 | 2 | 90% |
| `src/prompts.py` | 7 | 0 | 100% |
| `src/sandbox_runner.py` | 17 | 0 | 100% |
| `src/strip_thinking.py` | 17 | 0 | 100% |
| `training/rewards.py` | 4 | 0 | 100% |
| `training/sft_lora.py` | 48 | 15 | 69% |

**Lowest coverage:** `training/sft_lora.py` at 69%. The uncovered lines are the actual `transformers`/`peft`/`trl` model load and training loop, which execute only on GPU with real dependencies.

## What Each Group Protects

### `test_strip_thinking.py`
- Ensures regex stripping removes tags without corrupting code.
- Ensures balanced-tag detection catches mismatched open/close pairs.
- Ensures block counting is accurate for nested and multiline blocks.

### `test_sandbox_runner.py`
- Ensures code runs in an isolated temp file.
- Ensures timeouts are enforced (default 5s).
- Ensures temp files are cleaned up even on failure.

### `test_metrics.py`
- Ensures all six metrics are computed correctly.
- Ensures `executable_after_strip` is `False` for invalid syntax.

### `test_model_loader.py`
- Ensures base model loads via `transformers`.
- Ensures adapter path is forwarded to `peft.PeftModel`.
- Ensures missing `peft` raises `ImportError` when adapter is requested.

### `test_evaluator.py`
- Ensures mock mode produces valid JSONL with all required fields.
- Ensures real mode calls `load_model_and_tokenizer` and `model.generate`.
- Ensures adapter path is passed through to the loader.
- Ensures missing problems file raises `FileNotFoundError`.
- Ensures timeout is propagated to `run_code`.

### `test_pipeline.py`
- End-to-end integration: strip → sandbox → metrics on a sample problem.
- Dataset structural validation (fields, counts).
- SFT quality gates: 100% balanced tags, 100% strip-to-executable, ≥80% pass rate.

### `test_training_sft_lora.py`
- CLI argument parsing and defaults.
- Dry-run mode completes without loading a model.
- Missing dataset raises `FileNotFoundError`.

## Adding Tests

When adding new functionality:
1. Add tests in the appropriate `tests/test_*.py` file.
2. Run the full suite before committing.
3. Do not let total coverage drop below 89%.
