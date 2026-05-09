# Artifact Manifest - CodePause

This manifest lists all artifacts relevant to the CodePause Phase 7 submission.

## Model Adapter (Phase 6 Release Candidate)
- **Path**: `results/phase6_release_candidate/`
- **Rebuilt Zip**: `codepause_phase6_release_candidate_REBUILT.zip`
- **Files Included**:
  - `README_MODEL.md`: Documentation and loading instructions.
  - `metadata_model.json`: Structural metadata.
  - `eval_report.md`: Captured evaluation metrics.
  - `inference_smoke_report.md`: Smoke test logs.
  - `load_test_report.md`: Resource usage log.
  - `checksums.txt`: Integrity checks.
- **RECOVERY STATUS**: **Actual adapter binary files are reproducible via the Phase 6 Recovery cell in the primary notebook.** If the Colab runtime is reset, running that cell will recreate `results/phase6_release_candidate/` and the associated ZIP archive from the Phase 5 v5 recipe.

## Reports & Documentation
- **Submission README**: `submission/README_SUBMISSION.md`
- **Final Report**: `submission/FINAL_REPORT.md`
- **Demo Script**: `submission/DEMO_SCRIPT.md`
- **Model Card**: `submission/MODEL_CARD.md`
- **Results Summary**: `submission/results_summary.md`

## Training & Dataset Data
- **Held-out Problems**: `data/heldout_problems_30.jsonl`
- **SFT Dataset (v5)**: `data/thinkanywhere_sft_v5.jsonl`
- **Stripping Utility**: `src/strip_thinking.py`

## Notebooks
- **Primary Execution**: `notebooks/codepause_phase_5_dataset_v5_recovery_streaming.ipynb`
- **Phase 4 Failure Log**: `notebooks/codepause_phase_4_dataset_v4_laziness_loop.ipynb`

## Git Metadata
- **Branch**: `main`
- **State**: `READY_WITH_PENDING_ARTIFACT_SYNC` (Weights remain in Colab)
