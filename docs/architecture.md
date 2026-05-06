# Architecture

## Module Boundaries

| Module | Responsibility | Key Files |
|--------|---------------|-----------|
| `src/` | Reusable core logic | `strip_thinking.py`, `sandbox_runner.py`, `metrics.py`, `prompts.py`, `model_loader.py` |
| `eval/` | Evaluation orchestration | `evaluator.py`, `evaluate_baseline.py`, `evaluate_finetuned.py`, `make_report.py` |
| `training/` | Model training | `sft_lora.py`, `rewards.py` |
| `app/` | Local demo | `demo_mock.py` |
| `data/` | Datasets | `problems.jsonl`, `thinkanywhere_sft.jsonl`, `eval_cases.jsonl` |
| `tests/` | pytest suite | 13 test files, 95 tests |

## Data Flow

```text
problems.jsonl
       │
       ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  build_baseline │────▶│   model.generate   │────▶│  raw_output     │
│    _prompt()    │     │  (or mock)       │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────┐
                                               │ strip_thinking   │
                                               │ _blocks()        │
                                               └────────┬─────────┘
                                                        │
                                                        ▼
                                               ┌──────────────────┐
                                               │   run_code()     │
                                               │  (sandbox)       │
                                               └────────┬─────────┘
                                                        │
                                                        ▼
                                               ┌──────────────────┐
                                               │ compute_metrics()│
                                               └────────┬─────────┘
                                                        │
                                                        ▼
                                               ┌──────────────────┐
                                               │  results/*.jsonl │
                                               │  make_report.py  │
                                               └──────────────────┘
```

## Adapter-Aware Evaluation Design (Phase 0.5)

**Problem:** Phase 0 only supported baseline evaluation. Phase 1 requires evaluating a base model with an optional LoRA adapter without duplicating evaluator logic.

**Solution:**

1. **`src/model_loader.py`** — Unified loader that accepts `adapter_path`. If provided, wraps the base model with `peft.PeftModel`.
2. **`eval/evaluator.py`** — Shared `evaluate_model()` function used by both CLI entry points. Accepts `adapter_path` and forwards it to `load_model_and_tokenizer`.
3. **`eval/evaluate_baseline.py`** — Calls `evaluate_model(adapter_path=None)`.
4. **`eval/evaluate_finetuned.py`** — Calls `evaluate_model(adapter_path=<path>)`.

This design means:
- No duplication of generation, stripping, sandbox, or metric logic.
- Mock mode works identically for both baseline and fine-tuned evaluation.
- Adding new adapter types requires changes only in `model_loader.py`.

## Phase 0 Decisions

| Decision | Rationale |
|----------|-----------|
| Subprocess sandbox | Simple, no reliance on external execution services. Timeout enforced via `subprocess.run(timeout=...)`. |
| Regex tag stripping | Sufficient for well-formed tags. `re.DOTALL` handles multiline blocks. |
| JSONL datasets | Line-oriented, appendable, trivial to stream. |
| 30 problems / 30 SFT examples | Minimum viable for demonstration; scales by adding lines. |
| Gradio as optional demo | Falls back to text output if not installed, keeping dependencies minimal. |

## Phase 0.5 Decisions

| Decision | Rationale |
|----------|-----------|
| Extract `model_loader.py` | Isolates `transformers` + `peft` dependency from evaluation logic. |
| Shared `evaluate_model()` | Eliminates copy-paste between baseline and fine-tuned evaluators. |
| Mock mode in shared evaluator | Guarantees that baseline and adapter paths are tested without GPU. |
| `evaluate_finetuned.py` as thin CLI | Only parses args and delegates to `evaluator.py`. |

## Dependency Graph

```text
src/strip_thinking.py ──▶ src/metrics.py
src/sandbox_runner.py ──▶ src/metrics.py
src/prompts.py ─────────▶ eval/evaluator.py
src/model_loader.py ────▶ eval/evaluator.py
src/metrics.py ─────────▶ eval/evaluator.py
                          eval/evaluator.py ──▶ evaluate_baseline.py
                          eval/evaluator.py ──▶ evaluate_finetuned.py
eval/evaluator.py ───────▶ make_report.py (via JSONL)

training/sft_lora.py ────▶ transformers, peft, trl (runtime only)
training/rewards.py ─────▶ (stub, no runtime deps)
app/demo_mock.py ────────▶ src/* (optional gradio)
```

## Extension Points

- **New metric:** Add to `src/metrics.py`, update `compute_metrics()` signature if inputs are needed.
- **New model family:** Update `src/model_loader.py` device map / dtype defaults.
- **New evaluation mode:** Wrap `eval/evaluator.py` with a new thin CLI.
- **GRPO rewards:** Implement real `compute_rewards()` in `training/rewards.py` for Phase 2.
