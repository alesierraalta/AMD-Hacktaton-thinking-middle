# Colab T4 QLoRA Probe

## Overview

Phase 0.8 validates the 4-bit QLoRA training pipeline on Google Colab T4 (PCIe 15GB) before AMD MI300X GPU credit is approved.

**Hardware scope**: Google Colab T4 only. Results are **not** transferable to AMD MI300X.

## Hard Stops

A hard stop is a condition that halts the notebook before proceeding to the next step.

| Check | Condition | Action |
|-------|-----------|--------|
| GPU presence | `torch.cuda.is_available() == False` | Do not proceed — select GPU runtime |
| GPU type | "T4" not in GPU name | Only T4/A100/L4 are valid for QLoRA |
| OOM on load | `OutOfMemoryError` or "out of memory" in stderr | Reduce `max_seq_length` or switch to `local_colab_smoke` preset |
| Training loss NaN | "nan" in training output | Abort and report |
| Missing adapter | `adapter_config.json` absent after training | Do not proceed to evaluation |
| Pass rate < 50% | Mock evaluation pass rate below threshold | Flag for investigation before MI300X run |

## Drive Persistence Workflow

Session disconnection on Colab destroys all local state. Adapter and results **must** be written to Google Drive before the session ends.

Output path: `MyDrive/codepause/colab-t4-qlora/`

```
colab-t4-qlora/
├── adapter/              # LoRA adapter files
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── results/              # Evaluation outputs
│   ├── finetuned_colab_t4.jsonl
│   └── colab_t4_report.md
├── logs/
│   └── training_log.txt  # Captured stdout/stderr
└── session_meta.json     # Runtime metadata (GPU, status, timestamps)
```

**Rule**: Run the "Copy Outputs to Drive" cell last. If any checkpoint is missing, the cell raises `RuntimeError`.

## Preset Selection

| Preset | Model | VRAM | Use case |
|--------|-------|------|----------|
| Preset | Model | VRAM | Use case |
|--------|-------|------|----------|
| `local_colab_smoke` | Qwen2.5-Coder-0.5B-Instruct | ~6-8GB | Smoke test — validate adapter creation |
| `colab_primary_qwen` | Qwen2.5-Coder-1.5B-Instruct | ~10-12GB | Primary probe (this is the default) |
| `colab_non_qwen_granite` | Granite-3b-code-instruct-128k | ~10-12GB | Non-Qwen code probe |
| `colab_optional_gemma` | Gemma-4-E2B | ~10-12GB | Optional modern non-Qwen probe |
| `colab_stretch_qwen` | Qwen2.5-Coder-3B-Instruct | ~12-14GB | Stretch run after 0.5B and 1.5B succeed |

To override the default preset in the notebook, change the `--preset` argument in the Training cell:

```python
'--preset', 'local_colab_smoke',  # switch from colab_primary_qwen
```

## Notebook Sequence

1. **Drive Mount & Paths** — Set up output directories on Google Drive
2. **GPU Check** — HARD STOP if no T4/A100/L4 detected
3. **Install Dependencies** — `bitsandbytes==0.41.3.post2`, `peft`, `trl`, `transformers`
4. **Clone / Pull** — Pull latest CodePause from GitHub
5. **Smoke Training** — `sft_lora.py --smoke --load_in_4bit --max_steps 10 --limit_samples 8`
6. **Validate Adapter** — Check `adapter_config.json` and `adapter_model.safetensors` exist
7. **Mock Evaluation** — `evaluate_finetuned.py --mock --metadata_json {...}`
8. **Generate Report** — `make_report.py` for comparison
9. **Copy Outputs to Drive** — HARD STOP if any checkpoint is missing

## Expected Runtime

| Cell | Approximate time |
|------|-----------------|
| Install | ~3 minutes |
| Clone | ~30 seconds |
| Training (10 steps, 8 samples) | ~8-12 minutes |
| Evaluation (mock) | ~1 minute |
| Report | ~10 seconds |
| **Total** | **~15 minutes** |

## OOM Troubleshooting

If the Training cell raises an OOM error:

1. **Reduce `max_seq_length`** in the preset (e.g., from 2048 → 1024)
2. **Switch to `local_colab_smoke` preset** — lower memory footprint (~6-8GB)
3. **Reduce `per_device_train_batch_size`** to 1 and increase `gradient_accumulation_steps` accordingly
4. **Lower `--limit_samples`** to 4 samples instead of 8
5. **Check Colab GPU memory** — Colab free tier shows 15GB, but some of that is used by the notebook environment

```python
# Example: smoke preset fallback with reduced seq length
'--preset', 'local_colab_smoke',  # instead of colab_primary_qwen
'--max_seq_length', '1024',       # instead of 2048
```

## Session Disconnection

If Colab disconnects mid-training:

1. Reconnect and re-run from the beginning
2. The adapter output directory is already on Drive — training will skip already-saved files (TRL `save_strategy` is `"epoch"`)
3. Check `logs/training_log.txt` on Drive for the last checkpoint before disconnection

## Unsloth (Docs-Only Note)

> **Note**: [Unsloth](https://github.com/unsloth/unsloth) is a faster alternative to the standard `trl+peft` stack, achieving ~2× training speedup on T4. It is **not** a hard dependency of this phase — the TRL/PEFT stack is used to keep the pipeline consistent with Phase 1 (MI300X). If Colab T4 time is the bottleneck, Unsloth can be evaluated as a future optimization. Do not replace the training stack with Unsloth in this phase.

## Colab T4 vs. AMD MI300X

| Property | Colab T4 | AMD MI300X |
|----------|----------|------------|
| Precision | 4-bit nf4 (quantized) | BF16 / FP32 (full) |
| VRAM | 15GB | 256GB HBM |
| Training time | ~10-15 min (probe) | ~2-4h (full) |
| Purpose | Pipeline validation | Production training |
| Framework | CUDA / bitsandbytes | ROCm / HIP |

**This document does not claim MI300X results.** Phase 0.8 evidence is limited to Colab T4. MI300X execution will be documented in Phase 1.

## Pre-flight Checklist

Before opening the notebook, run locally:

```bash
python -m pytest tests/ -q
python eval/evaluate_baseline.py --mock --problems_path data/problems.jsonl --output_path results/baseline_mock.jsonl
python eval/evaluate_finetuned.py --mock --base_model mock-base --adapter_path mock-adapter --problems_path data/problems.jsonl --output_path results/finetuned_mock.jsonl
python eval/make_report.py --baseline results/baseline_mock.jsonl --finetuned results/finetuned_mock.jsonl --out results/mock_report.md
```

If any of the above fails, fix before running the Colab notebook.