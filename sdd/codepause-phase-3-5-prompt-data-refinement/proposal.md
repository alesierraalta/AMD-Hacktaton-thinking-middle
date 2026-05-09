# Phase 3.5 Proposal — Prompt & Data Refinement

## Change Name
`codepause-phase-3-5-prompt-data-refinement`

## Type
Refinement / Iteration on Phase 3

## Motivation
Phase 3 showed that ablation improvement is **prompt-heavy** — adapter does not outperform base+prompt (Arm C ≈ Arm B on mock). The hard gate for Phase 3.5 is that `adapter_v3 + best_prompt` must outperform `base + best_prompt`. This requires:
1. Promoting the best Phase 3 prompt to a stable named template
2. Creating a refined v3 dataset aligned with that prompt
3. Training adapter v3 with refined data
4. Evaluating to determine if adapter now contributes beyond prompt

## Scope
- Phase 3.5 SDD folder creation
- Prompt template promotion (`best_phase3_prompt` alias)
- `src/prompt_templates.py` update with new template
- Evaluator support for `--prompt_template best_phase3_prompt`
- Dataset v3 (`data/thinkanywhere_sft_v3.jsonl`) with ≥60 examples, exact signatures, executable code
- Dataset v3 validation report
- Tests for template rendering, evaluator support, dataset schema, strip/execution
- Recipe `config/recipes/phase3_5_qwen15b_prompt_data_refinement.yaml`
- Notebook `notebooks/codepause_phase_3_5_prompt_data_refinement.ipynb`
- Evaluation outputs: `results/phase3_5_base_best_prompt.jsonl`, `results/phase3_5_adapter_v3_best_prompt.jsonl`, `results/phase3_5_report.md`
- Archive with PASS/PARTIAL/FAIL based on hard gate

## Out of Scope
- Phase 3B runs
- Adding Granite/Gemma models
- GRPO integration
- Framework migration
- AMD result claims
- Test weakening
- Weak baseline comparisons

## Success Criteria
**Hard Gate:** `adapter_v3 + best_prompt > base + best_prompt` on held-out set (30 problems)

## Resources
- Best prompt: `thinkanywhere_qwen_instruct` from Phase 3 ablation
- Phase 2 v2 reference: 24/30 → 25/30 (+3.3pp)
- Held-out set: `data/heldout_problems_30.jsonl`
- Model: `Qwen/Qwen1.5-1.8B-Chat`
