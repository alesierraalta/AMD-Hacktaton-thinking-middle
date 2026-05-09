# Phase 3.5 Design — Prompt & Data Refinement

## Architecture

### Prompt Template Promotion
\est_phase3_prompt\ is an alias for \	hinkanywhere_qwen_instruct\:
- Defined in \src/prompt_templates.py\ as a new named template
- Registered in \PROMPT_TEMPLATE_REGISTRY\ and \get_prompt_template()\
- Supported in evaluators via existing \--prompt_template\ flag

### Evaluator Updates
1. \eval/evaluate_baseline.py\ — add \--prompt_template\ argument (uses \get_prompt_template()\)
2. \eval/evaluate_finetuned.py\ — add \--prompt_template\ argument
3. Both write \metadata.prompt_template_id\ to result JSONL for provenance

### Dataset v3 Design
- Path: \data/thinkanywhere_sft_v3.jsonl\
- Schema: \{instruction, response, problem_id}\ (v2 SFT schema)
- Minimum: 60 examples
- Source: Extend v2 with:
  - Edge case examples (empty input, boundary conditions)
  - Examples where \<thinkanywhere>\ appears near risk points
  - Correct implementations (not stubs)
  - Held-out problem-aligned function signatures

### Training Recipe
File: \config/recipes/phase3_5_qwen15b_prompt_data_refinement.yaml\

\\\yaml
model_name: "Qwen/Qwen1.5-1.8B-Chat"
dataset_path: "data/thinkanywhere_sft_v3.jsonl"
output_dir: "results/sft_phase3_5_v3"

lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

learning_rate: 1e-4
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
max_steps: 150
max_seq_length: 1024
epochs: 1
load_in_4bit: true
\\\

### Notebook Structure
\
otebooks/codepause_phase_3_5_prompt_data_refinement.ipynb\

Cells:
1. Drive/repo setup
2. Runtime check
3. Tests (prompt template, evaluator, dataset validation)
4. Dataset v3 validation
5. Base + best prompt evaluation
6. Adapter v3 training
7. Adapter v3 + best prompt evaluation
8. Report generation
9. Artifact copy to Drive

## Files to Create/Modify

| File | Action |
|------|--------|
| \src/prompt_templates.py\ | Add \est_phase3_prompt\ template |
| \eval/evaluate_baseline.py\ | Add \--prompt_template\ arg |
| \eval/evaluate_finetuned.py\ | Add \--prompt_template\ arg |
| \data/thinkanywhere_sft_v3.jsonl\ | Create (≥60 examples) |
| \esults/phase3_5_dataset_v3_quality_report.md\ | Create (dataset validation) |
| \config/recipes/phase3_5_qwen15b_prompt_data_refinement.yaml\ | Create |
| \
otebooks/codepause_phase_3_5_prompt_data_refinement.ipynb\ | Create |
| \esults/phase3_5_base_best_prompt.jsonl\ | Create (eval output) |
| \esults/phase3_5_adapter_v3_best_prompt.jsonl\ | Create (eval output) |
| \esults/phase3_5_report.md\ | Create (report output) |
| \	ests/test_prompt_templates.py\ | Create (template tests) |
| \	ests/test_evaluator_prompt_support.py\ | Create (evaluator CLI tests) |
| \	ests/test_dataset_v3.py\ | Create (dataset v3 tests) |

## Test Design

### test_prompt_templates.py
- Test \est_phase3_prompt\ renders correctly
- Test \get_prompt_template("best_phase3_prompt")\ returns \	hinkanywhere_qwen_instruct\ behavior
- Test registry includes \est_phase3_prompt\

### test_evaluator_prompt_support.py
- Test \evaluate_baseline.py --prompt_template best_phase3_prompt\ parses
- Test \evaluate_finetuned.py --prompt_template best_phase3_prompt\ parses

### test_dataset_v3.py
- Schema validation (instruction + response present)
- Minimum 60 examples
- Executable after strip
- Edge case coverage check
