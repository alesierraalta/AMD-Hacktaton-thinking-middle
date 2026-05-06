# Operations

## Cost Discipline

GPU credit on AMD Developer Cloud is finite. Follow these rules to avoid wasting it.

### Allowed GPU Usage

- Baseline inference (`eval/evaluate_baseline.py`)
- SFT/LoRA training (`training/sft_lora.py`)
- Fine-tuned evaluation (`eval/evaluate_finetuned.py`)

### Forbidden GPU Usage

- README edits
- Code refactoring
- Dataset writing or validation
- UI work
- Debugging unrelated local code
- Running the full test suite repeatedly (do this locally)

## GPU VM Rules

### Creation

- Use **MI300X**, 1 GPU.
- Select **Quick Start → PyTorch** image.
- Verify SSH access before starting any work.

### Container Access

```bash
ssh root@<PUBLIC_IP>
docker exec -it rocm bash
```

### Validation Before Training

Run these inside the container **before** any training:

```bash
rocm-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m pytest --cov=src --cov=training --cov=eval --cov=app --cov-report=term-missing
```

Expected:
- `torch.cuda.is_available()` is `True`.
- Device name identifies the AMD GPU.
- 95 tests pass, 91% coverage.

## Hard Stops

Stop immediately and do not continue spending GPU credit if any of the following occur:

- `torch.cuda.is_available()` is `False`.
- Tests fail before training starts.
- Baseline script crashes on more than 20% of examples.
- SFT fails before the first checkpoint.
- Disk usage exceeds safe threshold (risk of losing outputs).
- Outputs are not being pushed or copied externally.
- Adapter-aware evaluation readiness gate was not passed locally.

## Output Preservation

AMD Developer Cloud VMs are destroyed after use. **Outputs left only on the VM are lost.**

### Required Preservation

| Output | How to Preserve |
|--------|-----------------|
| Adapter weights | `git add` + commit + push, or `scp` / upload to cloud storage |
| Metrics JSONL | Same as above |
| Markdown report | Same as above |
| Screenshots | Upload to submission portal or cloud storage |

### VM Destruction

Do not just power off. AMD states that powered-off GPU VMs are still billed because resources remain reserved.

**Destroy the instance when done.**

## Operational Runbook

For step-by-step GPU execution, see [`sdd/codepause-phase-1-gpu-runbook.md`](../sdd/codepause-phase-1-gpu-runbook.md).

## Pre-GPU Checklist

- [ ] All 95 tests pass locally.
- [ ] Mock baseline evaluation produces 30 records.
- [ ] Mock fine-tuned evaluation produces 30 records.
- [ ] Report generation works from mock outputs.
- [ ] Git repo is clean and pushed.
- [ ] `sdd/codepause-phase-1-gpu-runbook.md` is in the repo.
