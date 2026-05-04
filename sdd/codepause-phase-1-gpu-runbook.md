# CodePause Phase 1 — GPU/ROCm Execution Runbook

## Goal

Run the first GPU-backed baseline and LoRA/SFT training cycle for CodePause on AMD Developer Cloud using MI300X + ROCm + PyTorch.

Phase 1 must prove the end-to-end path:

```text
baseline → SFT/LoRA → evaluation → report
```

GRPO is explicitly out of scope and remains Phase 2.

## Preconditions

Do not start GPU work until every item is true:

- [ ] Phase 0 complete.
- [ ] GitHub repo pushed.
- [ ] Tests passing locally.
- [ ] AMD Developer Cloud credit approved.
- [ ] GPU VM created with MI300X.
- [ ] Quick Start image selected: PyTorch.
- [ ] SSH access working.

## Readiness gate: adapter-aware evaluation

**GPU usage is BLOCKED until the following local verification passes.**

Run these commands in order. Every command must succeed before starting the GPU VM:

1. Full test suite:
   ```bash
   rtk python -m pytest --cov=src --cov=training --cov=eval --cov=app --cov-report=term-missing
   ```
   Expected: all tests pass, coverage >= 89%.

2. Mock baseline evaluation:
   ```bash
   rtk python eval/evaluate_baseline.py --mock --problems_path data/problems.jsonl --output_path results/baseline_mock.jsonl
   ```
   Expected: `results/baseline_mock.jsonl` created with 30 records.

3. Mock fine-tuned evaluation:
   ```bash
   rtk python eval/evaluate_finetuned.py --mock --base_model mock-base --adapter_path mock-adapter --problems_path data/problems.jsonl --output_path results/finetuned_mock.jsonl
   ```
   Expected: `results/finetuned_mock.jsonl` created with 30 records.

4. Comparison report generation:
   ```bash
   rtk python eval/make_report.py --baseline results/baseline_mock.jsonl --finetuned results/finetuned_mock.jsonl --out results/mock_report.md
   ```
   Expected: `results/mock_report.md` created containing "Baseline" and "Fine-tuned" sections.

**If any step fails, fix it locally before creating the GPU VM. Do not discover adapter evaluation gaps on paid GPU time.**

## Source baseline

Phase 0 archive:

- `sdd/codepause-phase-0/archive-report` #3219
- 22/22 tasks complete
- 71 tests passed
- 89% total coverage
- Critical warnings: 0

## Execution order

### 1. Create GPU VM

Use:

- GPU: MI300X, 1 GPU
- Image: Quick Start → PyTorch

AMD’s single-GPU fine-tuning guide says it covers “fine-tuning and inference techniques on a single-accelerator system”, which matches the first GPU pass for this project.

### 2. SSH into VM

```bash
ssh root@<PUBLIC_IP>
```

### 3. Enter ROCm container

```bash
docker exec -it rocm bash
```

AMD’s Developer Cloud guide references Quick Start environments and the ROCm/PyTorch path; AMD also documents ROCm fine-tuning with `transformers`, `datasets`, `huggingface-hub`, `peft`, `trl`, and `scipy`.

### 4. Clone repo

```bash
git clone <YOUR_REPO_URL>
cd codepause-amd
```

If the repo directory is still named `AMDh`, use that directory name instead. Do not rename during paid GPU time unless required.

### 5. Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

### 6. Validate GPU

```bash
rocm-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY
```

Expected:

- `torch.cuda.is_available()` prints `True`.
- Device count is at least `1`.
- Device name identifies the AMD GPU/MI300X environment.

### 7. Run test suite before training

```bash
rtk python -m pytest --cov=src --cov=training --cov=eval --cov=app --cov-report=term-missing
```

Expected from Phase 0:

```text
71 passed
89% coverage
```

### 8. Run baseline

```bash
python eval/evaluate_baseline.py \
  --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --problems_path data/problems.jsonl \
  --output_path results/baseline_gpu.jsonl
```

### 9. Run SFT/LoRA

```bash
python training/sft_lora.py \
  --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dataset_path data/thinkanywhere_sft.jsonl \
  --output_dir outputs/qwen25-coder-codepause-lora \
  --epochs 2 \
  --max_seq_length 4096
```

AMD’s ROCm fine-tuning documentation lists `peft` and `trl` as part of the fine-tuning environment, so this is aligned with the intended ROCm training stack.

### 10. Evaluate fine-tuned model

```bash
python eval/evaluate_finetuned.py \
  --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --adapter_path outputs/qwen25-coder-codepause-lora \
  --problems_path data/problems.jsonl \
  --output_path results/finetuned_gpu.jsonl
```

> **Phase 1 readiness note:** Adapter-aware evaluation must be verified locally via the Readiness gate above before creating the GPU VM.

### 11. Generate report

```bash
python eval/make_report.py \
  --baseline results/baseline_gpu.jsonl \
  --finetuned results/finetuned_gpu.jsonl \
  --out results/phase1_report.md
```

### 12. Preserve outputs

```bash
git add results/phase1_report.md
 git commit -m "Add Phase 1 GPU baseline and SFT results"
git push
```

Also upload or preserve externally:

- adapter output
- metrics JSONL
- report markdown
- screenshots for submission

### 13. Destroy VM when done

Do not just power it off. AMD states that powered-off GPU VMs are still billed because resources remain reserved, and “Charges are incurred until the instance is destroyed.”

## Success criteria

Phase 1 is complete when:

- [ ] GPU VM was created successfully.
- [ ] ROCm/PyTorch detected MI300X.
- [ ] Full Phase 0 test suite passed on the VM.
- [ ] Baseline results were generated.
- [ ] SFT/LoRA completed.
- [ ] Fine-tuned evaluation completed.
- [ ] Phase 1 report generated.
- [ ] VM destroyed or snapshot strategy documented.

## Hard stop conditions

Stop and do not continue spending GPU credit if:

- `torch.cuda.is_available()` is `False`.
- Tests fail before training.
- Baseline script crashes on more than 20% of examples.
- SFT fails before first checkpoint.
- Disk usage exceeds safe threshold.
- Repo outputs are not being pushed externally.
- Adapter-aware evaluation readiness gate not passed locally.

## Cost discipline

Use the GPU only for:

- baseline inference
- SFT/LoRA
- fine-tuned evaluation

Do not use GPU time for:

- README edits
- refactoring
- dataset writing
- UI work
- debugging unrelated local code

AMD states that if a GPU instance is powered off, it is still billed and credit hours still apply.

## Phase 2 boundary

Do not start GRPO in Phase 1. Phase 1 exists to demonstrate that the baseline → SFT/LoRA → evaluation → report pipeline works end-to-end on AMD MI300X.
