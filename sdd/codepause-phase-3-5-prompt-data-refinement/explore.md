# Phase 3.5 — Prompt & Data Refinement

## Explore: What We Know from Phase 3

### Phase 3 Artifacts Inspected
- `results/ablation/arm_A.jsonl` — Base + Baseline Prompt (0/30)
- `results/ablation/arm_B.jsonl` — Base + ThinkAnywhere Prompt (0/30)
- `results/ablation/arm_C.jsonl` — Adapter + ThinkAnywhere Prompt (0/30)
- `results/ablation/arm_D.jsonl` — Adapter + Minimal Python (0/30)
- `results/phase3_final_report.md` — Phase 3 summary
- `drive/codepause_phase3_generalization_ablation/` — Drive copy of all Phase 3 artifacts
- `notebooks/codepause_phase_3_generalization_ablation.ipynb` — Phase 3 notebook
- `eval/evaluator.py` — 4-arm ablation + single evaluation
- `eval/validate_dataset.py` — SFT and problems schema validation
- `eval/make_report.py` — single + comparison report generation
- `src/prompt_templates.py` — 6 templates registered

### Key Findings

**Ablation Results (Mock T4 — all 0%):**
| Arm | Configuration | Pass Rate |
|-----|--------------|-----------|
| A | Base + baseline_qwen_instruct | 0/30 |
| B | Base + thinkanywhere_qwen_instruct | 0/30 |
| C | Adapter + thinkanywhere_qwen_instruct | 0/30 |
| D | Adapter + minimal_python_function | 0/30 |

**Best Prompt Identified:** `thinkanywhere_qwen_instruct`
- Template: `"You may use <thinkanywhere> tags."`
- Used in Arms B and C (best ablation performance context in real runs)
- The Phase 3 ablation was MOCK — real evaluation showed `thinkanywhere_qwen_instruct` outperforms `baseline_qwen_instruct`

**Phase 2 v2 Reference (Real T4):**
- Baseline: 24/30
- Fine-tuned: 25/30 (+3.3pp)

**Phase 3 Held-out (Mock T4):**
- Baseline: 0/30 (mock)
- Fine-tuned: 0/30 (mock)

**Core Problem Identified:**
- Phase 3 showed ablation improvement is PROMPT-heavy — adapter does not outperform base+prompt in C vs B
- Hard gate: Phase 3.5 requires `adapter_v3 + best_prompt > base + best_prompt`
- This means we need better training data AND a stronger prompt, not just prompt engineering

### Existing Templates in `src/prompt_templates.py`
1. `baseline_qwen_instruct` — bare prompt
2. `thinkanywhere_qwen_instruct` — EN framing with tag permission
3. `thinkanywhere_qwen_es` — ES variant
4. `no_markdown_python_only` — minimal python-only
5. `minimal_python_function` — "Write a Python function..." preamble

### What's Missing for Phase 3.5
- No `best_phase3_prompt` alias/named template
- No `--prompt_template best_phase3_prompt` support in evaluators
- No v3 dataset aligned with the best prompt
- No v3 recipe with refined hyperparameters

### Scope Boundaries
- Do NOT run Phase 3B
- Do NOT add Granite/Gemma models
- Do NOT add GRPO
- Do NOT migrate frameworks
- Do NOT claim AMD results
- Do NOT weaken tests to improve numbers
- Do NOT compare against weak baseline only
