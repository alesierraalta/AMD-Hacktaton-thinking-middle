# Proposal: CodePause Phase 6 — Release Candidate Model

## Intent
Package the best current CodePause result (Phase 5) as a tested, documented, and reproducible release candidate adapter.

## Scope
- Recreate the missing Phase 5 adapter on Google Colab T4.
- Perform a load test to ensure the adapter is compatible with PEFT.
- Run an inference smoke test for basic function generation.
- Verify the evaluation metrics (+2/30 delta over baseline).
- Generate all required metadata and documentation artifacts.

## Approach
1.  **Recreation**: Execute `training/sft_lora.py` on Colab T4 using the `data/thinkanywhere_sft_v5.jsonl` dataset.
2.  **Collection**: Sync the resulting adapter from Colab to `results/phase6_release_candidate/`.
3.  **Verification**:
    - Load test: Load model + adapter using `src.model_loader`.
    - Smoke test: Generate code for `add`, `is_even`, `reverse_string`.
    - Eval: Rerun evaluation on 30 problems or verify against Phase 5 JSONL.
4.  **Packaging**: Generate `metadata_model.json`, `README_MODEL.md`, and `checksums.txt`.

## Hard Gates
- `adapter_config.json` and `adapter_model.safetensors` must exist.
- Adapter must load without error.
- Inference must complete.
- Metadata must be complete and accurate.

## Colab T4 Compatibility
Recreation will be performed on Tesla T4 to maintain the "Official CodePause claim environment".
VRAM requirement: ~10GB (QLoRA 4-bit).
Time estimate: ~10-15 minutes for 1 epoch.
