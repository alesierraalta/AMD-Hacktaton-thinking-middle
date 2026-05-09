# Phase 3.5 Apply Progress

## Status: Complete

Phase 3.5 implementation and Colab verification are complete. Final archive status: **PASS_WITH_LIMITATIONS**.

## Completed Tasks

### Phase 3.5.1: SDD Artifacts
- [x] Create `explore.md`
- [x] Create `proposal.md`
- [x] Create `spec.md`
- [x] Create `design.md`
- [x] Create `tasks.md`
- [x] Create `apply-progress.md`
- [x] Create `verify-report.md`
- [x] Create `archive-report.md`

### Phase 3.5.2: Prompt Template Promotion
- [x] Identified best prompt: `thinkanywhere_qwen_instruct` from Phase 3 ablation Arm B.
- [x] Promoted it as `best_phase3_prompt` in `src/prompt_templates.py`.
- [x] Confirmed evaluator CLIs accept `--prompt_template best_phase3_prompt`.

### Phase 3.5.3: Dataset v3
- [x] Built `data/thinkanywhere_sft_v3.jsonl`.
- [x] Included 61 examples (minimum was 60).
- [x] Covered conditionals, loops, indexing, parsing, boundary cases, recursion, and empty input cases.
- [x] Ensured code is executable after stripping thinking tags.
- [x] Generated `results/phase3_5_dataset_v3_quality_report.md`.

### Phase 3.5.4: Tests
- [x] Added prompt rendering tests.
- [x] Added evaluator prompt-template support tests.
- [x] Added dataset v3 schema/strip/execution tests.
- [x] Added Phase 3.5 report-generation test from mock outputs.

### Phase 3.5.5: Recipe and Notebook
- [x] Created `config/recipes/phase3_5_qwen15b_prompt_data_refinement.yaml`.
- [x] Created and repaired `notebooks/codepause_phase_3_5_prompt_data_refinement.ipynb`.
- [x] Added robust Colab repo/bundle setup.
- [x] Added dependency checks for `trl`, `bitsandbytes`, and `torchao>=0.16.0`.

### Phase 3.5.6: Colab T4 Evaluation
- [x] Ran base + best prompt evaluation: 3/30.
- [x] Trained adapter v3 with v3 dataset: completed at 150 steps.
- [x] Ran adapter v3 + best prompt evaluation: 5/30.
- [x] Generated `results/phase3_5_report.md`.
- [x] Copied artifacts to `/content/codepause_phase3_5`.

## Hard Gate

`adapter_v3 + best_prompt > base + best_prompt`

`5/30 > 3/30` -> **PASS**

## Issues Fixed During Apply/Verification

- Removed invalid notebook JSON control characters and BOM.
- Fixed Colab setup so the notebook runs from `/content/AMDh`, not bare `/content`.
- Added overlay bundle support because Phase 3.5 files were local/uncommitted and absent from the GitHub clone.
- Installed missing training dependencies (`trl`, `bitsandbytes`).
- Upgraded `torchao` to a PEFT-compatible version for adapter evaluation.

## Remaining Limitations

- The result is Colab T4 only; do not claim AMD results.
- Absolute score remains low at 5/30.
- Dataset v3 is small and should be refined before scaling.
- Adapter regressed some base+prompt successes, so v4 should target both persistent failures and regressions.
