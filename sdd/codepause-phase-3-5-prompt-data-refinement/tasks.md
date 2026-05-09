# Phase 3.5 Tasks — Prompt & Data Refinement

## Task Checklist

### Phase 3.5.1: SDD Artifacts
- [x] 3.5.1.1 Create `sdd/codepause-phase-3-5-prompt-data-refinement/explore.md`
- [x] 3.5.1.2 Create `sdd/codepause-phase-3-5-prompt-data-refinement/proposal.md`
- [x] 3.5.1.3 Create `sdd/codepause-phase-3-5-prompt-data-refinement/spec.md`
- [x] 3.5.1.4 Create `sdd/codepause-phase-3-5-prompt-data-refinement/design.md`
- [x] 3.5.1.5 Create `sdd/codepause-phase-3-5-prompt-data-refinement/tasks.md`
- [x] 3.5.1.6 Create `sdd/codepause-phase-3-5-prompt-data-refinement/apply-progress.md`
- [x] 3.5.1.7 Create `sdd/codepause-phase-3-5-prompt-data-refinement/verify-report.md`
- [x] 3.5.1.8 Create `sdd/codepause-phase-3-5-prompt-data-refinement/archive-report.md`

### Phase 3.5.2: Prompt Template Promotion
- [x] 3.5.2.1 Inspect Phase 3 ablation results to identify best prompt.
- [x] 3.5.2.2 Add `best_phase3_prompt` to `src/prompt_templates.py` as an alias.
- [x] 3.5.2.3 Register `best_phase3_prompt` in the prompt template registry.
- [x] 3.5.2.4 Ensure `get_prompt_template()` resolves `best_phase3_prompt`.

### Phase 3.5.3: Evaluator Updates
- [x] 3.5.3.1 Ensure `eval/evaluate_baseline.py` supports `--prompt_template`.
- [x] 3.5.3.2 Ensure `eval/evaluate_finetuned.py` supports `--prompt_template`.

### Phase 3.5.4: Dataset v3
- [x] 3.5.4.1 Build `data/thinkanywhere_sft_v3.jsonl` (61 examples).
- [x] 3.5.4.2 Ensure exact function signatures.
- [x] 3.5.4.3 Ensure code is executable after stripping `<think>` and `<thinkanywhere>` tags.
- [x] 3.5.4.4 Align examples with `best_phase3_prompt`.
- [x] 3.5.4.5 Include empty input, boundary, parsing, indexing, recursion, loops, and conditionals.
- [x] 3.5.4.6 Place `<thinkanywhere>` near real risk points.
- [x] 3.5.4.7 Generate `results/phase3_5_dataset_v3_quality_report.md`.

### Phase 3.5.5: Tests
- [x] 3.5.5.1 Add `tests/test_phase3_prompt_template.py`.
- [x] 3.5.5.2 Add `tests/test_phase3_evaluator_prompt_support.py`.
- [x] 3.5.5.3 Add `tests/test_phase3_dataset_v3.py`.
- [x] 3.5.5.4 Add `tests/test_phase3_5_report_generation.py`.

### Phase 3.5.6: Training Recipe
- [x] 3.5.6.1 Create `config/recipes/phase3_5_qwen15b_prompt_data_refinement.yaml`.

### Phase 3.5.7: Notebook
- [x] 3.5.7.1 Create `notebooks/codepause_phase_3_5_prompt_data_refinement.ipynb`.
- [x] 3.5.7.2 Fix notebook JSON validity.
- [x] 3.5.7.3 Add Colab repo/bundle setup.
- [x] 3.5.7.4 Add Colab dependency checks.

### Phase 3.5.8: Evaluation & Report
- [x] 3.5.8.1 Run base + best prompt evaluation -> `results/phase3_5_base_best_prompt.jsonl` (3/30).
- [x] 3.5.8.2 Train adapter v3 with v3 dataset -> `results/sft_phase3_5_v3`.
- [x] 3.5.8.3 Run adapter v3 + best prompt evaluation -> `results/phase3_5_adapter_v3_best_prompt.jsonl` (5/30).
- [x] 3.5.8.4 Generate `results/phase3_5_report.md`.

### Phase 3.5.9: Archive
- [x] 3.5.9.1 Determine PASS/PARTIAL/FAIL based on hard gate -> PASS.
- [x] 3.5.9.2 Write archive report with final status.
- [x] 3.5.9.3 Update SDD files with final state.

## Summary

- Completed: 31/31 tasks.
- Final hard gate: PASS (`5/30 > 3/30`).
- Archive status: PASS_WITH_LIMITATIONS.
