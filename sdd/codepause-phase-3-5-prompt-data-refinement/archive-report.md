# Phase 3.5 Archive Report

## Change
`codepause-phase-3-5-prompt-data-refinement`

## Final Status: PASS

## Hard Gate

**Requirement:** `adapter_v3 + best_prompt > base + best_prompt`

**Result:** PASS

| Run | Score |
|---|---:|
| Base + best prompt | 3/30 (10.0%) |
| Adapter v3 + best prompt | 5/30 (16.7%) |
| Delta | +2/30 (+6.7pp) |

The adapter now contributes beyond the prompt in the controlled Phase 3.5 comparison.

## Artifacts Created or Updated

### SDD
- `sdd/codepause-phase-3-5-prompt-data-refinement/explore.md`
- `sdd/codepause-phase-3-5-prompt-data-refinement/proposal.md`
- `sdd/codepause-phase-3-5-prompt-data-refinement/spec.md`
- `sdd/codepause-phase-3-5-prompt-data-refinement/design.md`
- `sdd/codepause-phase-3-5-prompt-data-refinement/tasks.md`
- `sdd/codepause-phase-3-5-prompt-data-refinement/apply-progress.md`
- `sdd/codepause-phase-3-5-prompt-data-refinement/verify-report.md`
- `sdd/codepause-phase-3-5-prompt-data-refinement/archive-report.md`

### Code/Data/Config
- `src/prompt_templates.py` — promoted `best_phase3_prompt` alias.
- `data/thinkanywhere_sft_v3.jsonl` — 61-example SFT dataset v3.
- `config/recipes/phase3_5_qwen15b_prompt_data_refinement.yaml` — controlled training recipe.
- `notebooks/codepause_phase_3_5_prompt_data_refinement.ipynb` — Colab T4 execution notebook with repo/bundle setup and dependency checks.
- `scripts/create_phase3_5_bundle.py` — utility for creating the Colab overlay bundle.

### Tests
- `tests/test_phase3_prompt_template.py`
- `tests/test_phase3_evaluator_prompt_support.py`
- `tests/test_phase3_dataset_v3.py`
- `tests/test_phase3_5_report_generation.py`

### Results
- `results/phase3_5_dataset_v3_quality_report.md`
- `results/phase3_5_base_best_prompt.jsonl`
- `results/phase3_5_adapter_v3_best_prompt.jsonl`
- `results/phase3_5_report.md`
- `results/sft_phase3_5_v3` (Colab artifact copied to `/content/codepause_phase3_5`; large adapter files are not expected to be committed unless explicitly desired)

## Best Prompt

- Template ID: `best_phase3_prompt`
- Source template: `thinkanywhere_qwen_instruct`
- Prompt: `You may use <thinkanywhere> tags.`
- Source artifact: Phase 3 ablation Arm B.

## Validation Summary

- Prompt/evaluator/dataset tests: 17/17 passed on Colab.
- Dataset validation: PASSED (61 examples, 100% `<thinkanywhere>` coverage).
- Base + best prompt evaluation: 3/30.
- Adapter v3 training: completed, `max_steps=150`.
- Adapter v3 + best prompt evaluation: 5/30.
- Report generation: completed.

## Comparison with Prior Phases

| Phase | Configuration | Result | Delta |
|---|---|---:|---:|
| Phase 2 v2 | base -> fine-tuned | 24/30 -> 25/30 | +3.3pp |
| Phase 3 real T4 | baseline -> fine-tuned | 4/30 -> 8/30 | +13.3pp |
| Phase 3.5 real T4 | base+best_prompt -> adapter_v3+best_prompt | 3/30 -> 5/30 | +6.7pp |

Do not use the unsupported old Phase 2 claim (23/30 -> 26/30) unless its artifact set is recovered.

## Limitations

1. Result is from Colab T4, not AMD hardware.
2. Absolute pass rate remains low (5/30).
3. Dataset v3 is still small (61 examples).
4. Adapter improved the aggregate score but still regressed some examples that base+prompt passed.
5. Phase 3.5 validates adapter contribution beyond prompt; it does not prove readiness for scaling without failure analysis.

## Next Recommended Step

Do targeted dataset refinement before model scaling:

1. Analyze per-problem flips in `results/phase3_5_base_best_prompt.jsonl` vs `results/phase3_5_adapter_v3_best_prompt.jsonl`.
2. Add v4 examples for regressions and persistent failures.
3. Re-run the same hard gate.
4. Scale model only if the adapter contribution remains positive and regression count drops.

## Archive Decision

Archive Phase 3.5 as **PASS_WITH_LIMITATIONS**: the hard gate passed, but the low absolute score requires further data refinement before any scaling claim.
