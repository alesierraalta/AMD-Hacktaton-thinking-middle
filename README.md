# CodePause: Think-Anywhere Code Generation on AMD

## Release Status

CodePause is ready for final submission.

The project is a Colab T4 reproducible, Think-Anywhere-inspired LoRA/QLoRA prototype. It implements the cold-start/SFT side of the Think-Anywhere idea by training an adapter to emit removable `<thinkanywhere>` reasoning blocks during code generation, then strips those blocks before execution and evaluates the resulting Python code with deterministic tests.

This is not a full reproduction of the original paper: RLVR, original benchmark scale, and original hardware scale are out of scope.

Final adapter integrity has been verified:

`results/phase6_release_candidate/adapter_model.safetensors`

SHA256:

`34e0c6895d15850ef3227424edce57aaa0dc67f0257c24bec969afe72fc77e80`

Status:

`READY_FOR_FINAL_SUBMISSION ✅`

## Problem

This project implements the "Think Anywhere" mechanism for code generation on AMD hardware. It enables LLMs to invoke reasoning during code generation via special `<think>` and `<thinkanywhere>` tags, producing executable code after stripping these thinking blocks.

## Solution

A two-stage pipeline:
1. **Cold-start training** (SFT/LoRA) with Think-Anywhere examples
2. **RLVR reinforcement** (Phase 2, deferred to GPU phase)

## Pipeline

```
Prompt → Tagged Output → Strip Thinking Blocks → Sandbox Execution → Metrics
```

- **Prompt**: Problem description formatted for the model
- **Tagged Output**: Model response containing `<think>` and `<thinkanywhere>` blocks
- **Strip**: Remove thinking tags to obtain executable code
- **Sandbox**: Run generated code against problem tests with timeout enforcement
- **Metrics**: Compute pass rate, tag balance, block count, executability, line count

## Repo Structure

```
codepause-amd/
├── README.md
├── requirements.txt
├── data/              # Datasets (problems, SFT examples, eval cases)
│   ├── problems.jsonl
│   ├── thinkanywhere_sft.jsonl
│   └── eval_cases.jsonl
├── src/               # Core utilities (strip, sandbox, metrics, prompts)
│   ├── strip_thinking.py
│   ├── sandbox_runner.py
│   ├── metrics.py
│   └── prompts.py
├── training/          # SFT/LoRA scripts
│   ├── sft_lora.py
│   └── rewards.py
├── eval/              # Baseline evaluation and reporting
│   ├── evaluate_baseline.py
│   └── make_report.py
├── app/               # Local demo
│   └── demo_mock.py
├── results/           # Outputs and metrics
└── tests/             # pytest suite
```

## Local Tests

Install dependencies and run the test suite:

```bash
pip install -r requirements.txt
python -m pytest
```

### Verification Commands

```bash
# Dry-run SFT script (no GPU required)
python training/sft_lora.py \
  --model_name mock \
  --dataset_path data/thinkanywhere_sft.jsonl \
  --output_dir results/sft \
  --dry-run

# Mock baseline evaluation (no model required)
python eval/evaluate_baseline.py \
  --model_name mock \
  --problems_path data/problems.jsonl \
  --output_path results/baseline.jsonl \
  --mock

# Generate report from baseline results
python eval/make_report.py --input_path results/baseline.jsonl

# Run mock demo
python app/demo_mock.py
```

## GPU Execution Plan (Official: Google Colab T4)

The official execution environment for CodePause Phase 1 is **Google Colab T4**.

1. **Open the official Colab Notebook**: `notebooks/codepause_phase_1c_colab_only_qlora.ipynb`
2. **Select T4 GPU** runtime in Google Colab.
3. **Mount Google Drive** to preserve outputs.
4. **Run the Notebook** which automatically:
   - Clones the repository
   - Installs dependencies
   - Runs the local test suite
   - Trains the Qwen 0.5B smoke model (QLoRA)
   - Evaluates the baseline vs fine-tuned 0.5B model
   - Trains the Qwen 1.5B primary model (QLoRA)
   - Evaluates the baseline vs fine-tuned 1.5B model
   - Copies all results to Google Drive

### Historical/Optional: AMD MI300X Execution Plan

*Note: AMD MI300X is currently deprecated as the official path unless future credits appear.*

When AMD Developer Cloud access is approved:
1. **Create MI300X GPU Droplet** via AMD Developer Cloud portal
2. **Select Quick Start → PyTorch** (ROCm-enabled container)
3. **SSH into the VM** and access the container: `docker exec -it rocm bash`
4. **Clone this repo** and install requirements.
5. **Run training/evaluation** using the local CLI scripts.

## Metrics

The pipeline computes six metrics per example:

| Metric | Description |
|--------|-------------|
| `tests_passed` | Whether the generated code passed all problem tests |
| `balanced_tags` | Whether all `<think>` / `<thinkanywhere>` tags are balanced |
| `has_thinkanywhere` | Whether the output contains at least one `<thinkanywhere>` block |
| `thinkanywhere_blocks` | Count of `<thinkanywhere>` blocks |
| `executable_after_strip` | Whether the stripped code is syntactically valid Python |
| `clean_code_lines` | Number of lines in the stripped code |

## Acceptance Criteria

- Tag balance: 100% of SFT examples have balanced tags
- Strip success: 100% of SFT examples strip to executable code
- Local test pass rate: ≥80%
