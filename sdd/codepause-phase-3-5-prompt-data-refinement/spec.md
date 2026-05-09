# Phase 3.5 Specification — Prompt & Data Refinement

## Overview
Phase 3.5 implements prompt template promotion, dataset v3 creation, and controlled training
to determine if adapter_v3 + best_prompt outperforms base + best_prompt on held-out evaluation.

## Best Prompt Identification
- **Source:** Phase 3 ablation — 	hinkanywhere_qwen_instruct (Arm B / Arm C)
- **Template ID to promote:** est_phase3_prompt
- **Prompt text:** \You may use <thinkanywhere> tags.\
- **Rationale:** Phase 3 showed improvement is prompt-heavy. In real T4 runs, \	hinkanywhere_qwen_instruct\
  outperforms \aseline_qwen_instruct\. Arm B and C used this prompt.

## Promoted Template: est_phase3_prompt
Alias for \	hinkanywhere_qwen_instruct\ — same behavior, stable name for Phase 3.5 evaluation.

## Dataset v3 Requirements

### Schema
\\\json
{
  "instruction": "string (required, min 2 chars)",
  "response": "string (required, min 2 chars)",
  "problem_id": "string (optional but recommended)"
}
\\\

### Size
- Minimum: 60 examples

### Quality
1. **Exact function signatures** — entry points must match the held-out problem signatures
2. **Executable code after strip** — removing \<thinkanywhere>\ and \</think>\ tags produces valid Python
3. **No unnecessary Markdown** — evaluator expects code only
4. **Aligned with best prompt** — examples teach correct code, not tag placement
5. **Edge cases** — similar to Phase 3 failure patterns:
   - Empty input handling
   - Boundary conditions (indexing, loops)
   - Conditional logic near risk points
   - Recursion depth cases
   - Parsing edge cases

### Alignment with Phase 3 Failures
Phase 3 failures showed patterns:
- Stub functions with no body (pass-only implementations)
- Functions ignoring parameters
- Missing edge case handling

Dataset v3 must:
- Include examples where \<thinkanywhere>\ appears near real risk points
- Teach actual implementation logic, not just tag balancing
- Cover conditionals, loops, indexing, parsing, boundary cases, recursion, empty input

## Evaluation Scenarios

### A: Base + Best Prompt (Held-out)
\\\ash
rtk python eval/evaluate_baseline.py \\
  --model_name Qwen/Qwen1.5-1.8B-Chat \\
  --problems_path data/heldout_problems_30.jsonl \\
  --output_path results/phase3_5_base_best_prompt.jsonl \\
  --prompt_template best_phase3_prompt
\\\

### B: Adapter v3 + Best Prompt (Held-out)
\\\ash
rtk python eval/evaluate_finetuned.py \\
  --base_model Qwen/Qwen1.5-1.8B-Chat \\
  --adapter_path results/sft_phase3_5_v3 \\
  --problems_path data/heldout_problems_30.jsonl \\
  --output_path results/phase3_5_adapter_v3_best_prompt.jsonl \\
  --prompt_template best_phase3_prompt
\\\

## Hard Gate
\dapter_v3 + best_prompt > base + best_prompt\ — must show adapter contribution beyond prompt.
If not achieved: archive as PARTIAL or FAIL, recommend more dataset refinement before model scaling.

## Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|--------------|
| 1 | \est_phase3_prompt\ template exists in \src/prompt_templates.py\ | Import test |
| 2 | Evaluator supports \--prompt_template best_phase3_prompt\ | CLI test |
| 3 | Dataset v3 has ≥60 examples | \alidate_dataset.py --min_examples 60\ |
| 4 | Dataset v3 validates (SFT schema) | \alidate_dataset.py --schema sft\ |
| 5 | Code executable after strip | Per-example validation in v3 report |
| 6 | Base + best prompt evaluation completes | JSONL output exists |
| 7 | Adapter v3 training completes | Adapter files in \esults/sft_phase3_5_v3/\ |
| 8 | Adapter v3 + best prompt evaluation completes | JSONL output exists |
| 9 | Report generated | \esults/phase3_5_report.md\ exists |
| 10 | **Hard gate: adapter > base** | Delta > 0 |

## Phase 3 Comparison Reference
- Phase 2 v2 real result: 24/30 → 25/30 (+3.3pp)
- Phase 3 held-out (mock): 0/30 → 0/30
- Phase 3.5 goal: demonstrate adapter contribution beyond prompt alone
