# Explore: CodePause Phase 6 — Release Candidate Model

## Context
Phase 5 achieved a +2/30 delta (3/30 vs 1/30) and passed the hard gate. This is the best current controlled result.
Phase 6 objective is to package this result as a loadable and tested release candidate artifact.

## Findings

### Phase 5 Artifact Status
- **Results found**: `results/phase5_adapter_v5_best_prompt.jsonl` and `results/phase5_report.md` exist and confirm the 3/30 score.
- **Adapter missing**: `adapter_config.json` and `adapter_model.safetensors` for Phase 5 are missing from both the local filesystem and the current Colab environment.
- **Dataset found**: `data/thinkanywhere_sft_v5.jsonl` exists and is ready for use.
- **Training script ready**: `training/sft_lora.py` is verified to support the v5 dataset.

### Release Candidate Requirements
- Target directory: `results/phase6_release_candidate/`
- Required files:
  - Adapter files (`adapter_config.json`, `adapter_model.safetensors`)
  - Metadata (`metadata_model.json`)
  - Reports (`load_test_report.md`, `inference_smoke_report.md`, `eval_report.md`)
  - Documentation (`README_MODEL.md`)
  - Verification (`checksums.txt`)

### Recreation Strategy
Since the Phase 5 adapter is missing, it must be recreated using the verified v5 dataset and training configuration.
- **Environment**: Must use Google Colab T4 for official provenance.
- **Configuration**:
  - Model: `Qwen/Qwen1.5-1.8B-Chat`
  - Dataset: `data/thinkanywhere_sft_v5.jsonl`
  - Hyperparameters: LoRA rank 8, alpha 16, lr 1e-4 (matching Phase 3.5 successful run).
- **Provenance**: Artifacts will be tagged as Phase 5 source, Phase 6 recreation.

## Risks
- **Training failure**: Recreation might fail due to quota or environmental issues.
- **Metric drift**: Recreated model might achieve a different score than the original Phase 5 run (though delta should remain positive).
- **Environment constraints**: Running on Colab via MCP might be slower than direct UI.

## Recommendation
Proceed with recreation of Phase 5 adapter on Colab T4, then package as Phase 6 RC.
