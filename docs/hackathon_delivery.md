# Hackathon Delivery — CodePause Think-Anywhere

CodePause demonstrates **thinking-in-the-middle for code generation**: the model emits `<thinkanywhere>` reasoning blocks inside generated code, the evaluator strips those blocks, executes the remaining Python in a sandbox, and reports pass/fail metrics. The current verified result is a **minimal controlled improvement** on Colab T4: adapter v5 beats the same base model and same prompt baseline.

## One-line verdict

**It works as a measured prototype, not as a production-ready coding model yet.**

| Run | Result |
| --- | --- |
| Base + `best_phase3_prompt` | `1/30` |
| Adapter v5 + `best_phase3_prompt` | `3/30` |
| Delta | `+2/30` |
| Hard gate | `PASS` |

Evidence: `results/phase5_report.md`.

## Exact delivery route

### Demo route — live streaming

Use this route if the judges need to **see the model streaming output**.

1. Open Colab with T4 GPU.
2. Upload/open: `notebooks/codepause_phase_5_dataset_v5_recovery_streaming.ipynb`.
3. Run sections in order:
   - `1. Setup`
   - `2. Tests`
   - `3. Dependencies`
   - `5. Train adapter v5` if `results/sft_phase5_v5` is not already restored in the runtime
   - `8. Live streaming demo — Adapter v5`
4. Expected behavior:
   - The model loads `Qwen/Qwen1.5-1.8B-Chat`.
   - PEFT loads `results/sft_phase5_v5`.
   - `TextIteratorStreamer` prints generated tokens live.
   - Generated code may include `<thinkanywhere>...</thinkanywhere>` blocks.

### Reproducibility route — hard gate

Use this route if the judges need to verify the measured result.

1. Open: `notebooks/codepause_phase_5_dataset_v5_recovery_streaming.ipynb`.
2. Run sections:
   - `1. Setup`
   - `2. Tests`
   - `3. Dependencies`
   - `4. Baseline — base + best prompt`
   - `5. Train adapter v5`
   - `6. Evaluate adapter v5`
   - `7. Report + hard gate`
3. Expected result:
   - `results/phase5_base_best_prompt.jsonl`
   - `results/phase5_adapter_v5_best_prompt.jsonl`
   - `results/phase5_report.md`
   - hard gate prints `PASS` when adapter score is greater than baseline.

### Local route — no GPU

Use this route when GPU quota is gone.

```bash
python -m pytest tests/test_generate_v5.py tests/test_training_normalization.py -v
python eval/make_report.py \
  --baseline results/phase5_base_best_prompt.jsonl \
  --finetuned results/phase5_adapter_v5_best_prompt.jsonl \
  --out results/phase5_report.md
```

This validates local code/data/reporting, but it does **not** retrain or regenerate model outputs.

## Exact architecture

```text
Problem dataset
  data/heldout_problems_30.jsonl
        │
        ▼
Prompt template
  src/prompts.py::best_phase3_prompt
        │
        ▼
Model generation
  Base: Qwen/Qwen1.5-1.8B-Chat
  Adapter: results/sft_phase5_v5 via PEFT LoRA
        │
        ▼
Tagged output
  Python code + optional <thinkanywhere>...</thinkanywhere>
        │
        ▼
Strip thinking blocks
  src/strip_thinking.py
        │
        ▼
Sandbox execution
  src/sandbox_runner.py
        │
        ▼
Metrics
  src/metrics.py
        │
        ▼
Evaluation JSONL
  results/phase5_base_best_prompt.jsonl
  results/phase5_adapter_v5_best_prompt.jsonl
        │
        ▼
Report
  eval/make_report.py -> results/phase5_report.md
```

## Module map

| Area | Files | Responsibility |
| --- | --- | --- |
| Core prompt/strip/metrics | `src/prompts.py`, `src/strip_thinking.py`, `src/metrics.py` | Prompt selection, tag stripping, metric computation. |
| Safe execution | `src/sandbox_runner.py` | Runs generated Python with timeout and captures pass/fail. |
| Model loading | `src/model_loader.py` | Loads base model and optional PEFT adapter. |
| Evaluation | `eval/evaluator.py`, `eval/evaluate_baseline.py`, `eval/evaluate_finetuned.py` | Shared evaluation path for baseline and adapter. |
| Reporting | `eval/make_report.py` | Compares baseline vs adapter JSONL outputs. |
| Training | `training/sft_lora.py` | QLoRA/SFT training; supports `text`, `instruction/response`, and `prompt/raw_output` datasets. |
| Dataset v5 generation | `scripts/generate_sft_v5.py` | Creates high-entropy SFT data and rejects lazy code (`pass`, `TODO`, placeholders). |
| Tests | `tests/test_generate_v5.py`, `tests/test_training_normalization.py` | Guards generator quality and training schema normalization. |
| Notebooks | `notebooks/codepause_phase_5_dataset_v5_recovery_streaming.ipynb` | Colab T4 execution and streaming demo. |

## Training architecture

| Decision | Value |
| --- | --- |
| Base model | `Qwen/Qwen1.5-1.8B-Chat` |
| Fine-tuning | PEFT LoRA / QLoRA |
| Quantization | bitsandbytes 4-bit NF4 |
| Adapter output | `results/sft_phase5_v5` |
| Dataset | `data/thinkanywhere_sft_v5.jsonl` |
| Dataset size | 150 examples |
| Prompt for controlled eval | `best_phase3_prompt` |
| Official measured runtime | Google Colab T4 |

## What to say in the presentation

> We built a full evaluation loop for thinking-in-the-middle code generation. The model can emit `<thinkanywhere>` blocks inside code, we strip them before execution, then sandbox and score the remaining Python. Phase 5 recovered from a failed dataset direction and produced a controlled improvement: the adapter beats the same base model with the same prompt, 3/30 vs 1/30. The result is a working prototype signal, not a production claim.

## What not to claim

- Do **not** claim these are AMD hardware results unless rerun on AMD hardware.
- Do **not** claim production robustness; absolute score is still low.
- Do **not** claim Phase 4 worked; Phase 4 failed and motivated Phase 5.

## Files to show judges

| Need | File |
| --- | --- |
| Main demo notebook | `notebooks/codepause_phase_5_dataset_v5_recovery_streaming.ipynb` |
| Final report | `results/phase5_report.md` |
| Adapter eval output | `results/phase5_adapter_v5_best_prompt.jsonl` |
| Baseline eval output | `results/phase5_base_best_prompt.jsonl` |
| Dataset | `data/thinkanywhere_sft_v5.jsonl` |
| Dataset generator | `scripts/generate_sft_v5.py` |
| Training script | `training/sft_lora.py` |
| Architecture reference | `docs/architecture.md` |
| This delivery guide | `docs/hackathon_delivery.md` |

## Next step after delivery

Phase 6 should scale dataset v5 to 500+ examples and expand algorithm families before claiming stronger generalization.
