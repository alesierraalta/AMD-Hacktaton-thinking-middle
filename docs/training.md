# Training

## Overview

Training uses Supervised Fine-Tuning (SFT) with LoRA adapters to teach the model to emit Think-Anywhere tags during code generation. The script is `training/sft_lora.py`.

GRPO (reinforcement learning) is explicitly out of scope for Phase 1 and remains Phase 2.

## SFT/LoRA Script

**File:** `training/sft_lora.py`

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | required | Base model name or path |
| `--dataset_path` | required | Path to SFT dataset (JSONL) |
| `--output_dir` | `results/sft` | Directory for adapter output |
| `--epochs` | `1` | Training epochs |
| `--max_seq_length` | `1024` | Max sequence length |
| `--learning_rate` | `2e-5` | Learning rate |
| `--lora_rank` | `8` | LoRA rank |
| `--gradient_accumulation_steps` | `1` | Gradient accumulation steps |
| `--dry-run` | flag | Validate config and dataset without training |

## Dry-Run Mode (Local, No GPU)

Dry-run validates that the dataset loads and the configuration is sane without importing the model or starting training.

```bash
python training/sft_lora.py \
  --model_name mock \
  --dataset_path data/thinkanywhere_sft.jsonl \
  --output_dir results/sft \
  --dry-run
```

**What it checks:**
- Dataset file exists and is valid JSONL.
- All CLI arguments are parsed correctly.
- Prints config summary and dataset length.

## Local Execution (CPU)

Not recommended for full training due to speed, but possible for tiny models or debugging:

```bash
python training/sft_lora.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --dataset_path data/thinkanywhere_sft.jsonl \
  --output_dir results/sft \
  --epochs 1 \
  --max_seq_length 512
```

## GPU Execution (Google Colab T4)

Run inside the official Colab notebook `codepause_phase_1c_colab_only_qlora.ipynb` on a T4 GPU runtime.

*(Historical: Previously targeted for ROCm PyTorch container on AMD Developer Cloud.)*

```bash
python training/sft_lora.py \
  --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dataset_path data/thinkanywhere_sft.jsonl \
  --output_dir outputs/qwen25-coder-codepause-lora \
  --epochs 2 \
  --max_seq_length 4096
```

**LoRA configuration (hardcoded in script):**

| Parameter | Value |
|-----------|-------|
| `r` | `--lora_rank` (default 8) |
| `lora_alpha` | `rank * 2` |
| `target_modules` | `["q_proj", "v_proj"]` |
| `lora_dropout` | `0.05` |
| `bias` | `"none"` |
| `task_type` | `CAUSAL_LM` |

**Training arguments:**

| Parameter | Value |
|-----------|-------|
| `per_device_train_batch_size` | `1` |
| `save_strategy` | `"epoch"` |
| `logging_steps` | `10` |

## Adapter Output

After training completes, the output directory contains:

```text
outputs/qwen25-coder-codepause-lora/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files
```

These artifacts are consumed by `eval/evaluate_finetuned.py` via the `--adapter_path` argument.

## GRPO (Phase 2)

GRPO reinforcement learning is **not** implemented in Phase 1.

- `training/rewards.py` exists as a stub with placeholder reward computation.
- Real reward shaping (accuracy, format, tag balance) will be added in Phase 2.
- Do not attempt GRPO training during Phase 1 GPU time.

## Training Checklist

Before GPU training:
- [ ] Dry-run passes with the intended dataset.
- [ ] Dataset has at least 30 examples.
- [ ] All SFT examples have balanced tags and strip to valid Python.
- [ ] `output_dir` is inside the repo or a mounted volume so artifacts are preserved.
- [ ] `lora_rank` and `max_seq_length` fit in GPU memory for the chosen model.

After training:
- [ ] Adapter files exist in `--output_dir`.
- [ ] Adapter can be loaded by `eval/evaluate_finetuned.py --mock` (dry-run eval).
- [ ] Adapter output is committed or copied off the VM before destruction.
