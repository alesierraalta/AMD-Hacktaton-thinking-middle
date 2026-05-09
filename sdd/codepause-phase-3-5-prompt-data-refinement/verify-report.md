# Phase 3.5 Verify Report

## Status: PASS

Phase 3.5 has been verified on **Colab T4**. The hard gate passed:

```text
adapter_v3 + best_prompt > base + best_prompt
5/30 > 3/30
```

## Verification Evidence

### Setup
- Runtime: Colab T4 (`GPU: Tesla T4`)
- Repo root: `/content/AMDh`
- Phase 3.5 overlay bundle applied when GitHub checkout did not contain local uncommitted artifacts.

### Tests

```bash
python -m pytest tests/test_phase3_prompt_template.py tests/test_phase3_evaluator_prompt_support.py tests/test_phase3_dataset_v3.py -v
```

Result: **17/17 passed**

### Dataset v3 Validation

```bash
python eval/validate_dataset.py data/thinkanywhere_sft_v3.jsonl --schema sft --min_examples 60
```

Result: **PASSED**

- Dataset: `data/thinkanywhere_sft_v3.jsonl`
- Total examples: 61
- `<thinkanywhere>` coverage: 61/61 (100%)
- Schema: SFT (`instruction`, `response`)

### Best Prompt
- Template ID: `best_phase3_prompt`
- Alias/source template: `thinkanywhere_qwen_instruct`
- Prompt text: `You may use <thinkanywhere> tags.`
- Source: Phase 3 ablation Arm B (`results/ablation/arm_B.jsonl`)

## Required Evaluations

### A. Base + best prompt

```bash
python eval/evaluate_baseline.py \
  --model_name Qwen/Qwen1.5-1.8B-Chat \
  --problems_path data/heldout_problems_30.jsonl \
  --output_path results/phase3_5_base_best_prompt.jsonl \
  --prompt_template best_phase3_prompt
```

Result: **3/30 passed (10.0%)**

### B. Adapter v3 training

```bash
python training/sft_lora.py \
  --model_name Qwen/Qwen1.5-1.8B-Chat \
  --dataset_path data/thinkanywhere_sft_v3.jsonl \
  --output_dir results/sft_phase3_5_v3 \
  --epochs 1 \
  --max_steps 150 \
  --max_seq_length 1024 \
  --device auto \
  --load_in_4bit \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05
```

Result: **completed**

- Output: `results/sft_phase3_5_v3`
- Runtime: 672.5s
- Train loss: 1.056
- Final epoch: 18.79

### C. Adapter v3 + same best prompt

```bash
python eval/evaluate_finetuned.py \
  --base_model Qwen/Qwen1.5-1.8B-Chat \
  --adapter_path results/sft_phase3_5_v3 \
  --problems_path data/heldout_problems_30.jsonl \
  --output_path results/phase3_5_adapter_v3_best_prompt.jsonl \
  --prompt_template best_phase3_prompt
```

Result: **5/30 passed (16.7%)**

### D. Report generation

```bash
python eval/make_report.py \
  --baseline results/phase3_5_base_best_prompt.jsonl \
  --finetuned results/phase3_5_adapter_v3_best_prompt.jsonl \
  --out results/phase3_5_report.md
```

Result: **generated**

## Hard Gate

| Comparison | Score |
|---|---:|
| Base + best prompt | 3/30 |
| Adapter v3 + best prompt | 5/30 |
| Delta | +2/30 (+6.7pp) |

**Verdict: PASS** — adapter v3 contributes beyond the best prompt.

## Comparison Against Prior Phases

- Phase 2 v2 real result: 24/30 -> 25/30 (+3.3pp). This is the valid Phase 2 caveat.
- Phase 3 real T4 held-out result: baseline 4/30 -> fine-tuned 8/30 (+13.3pp), but ablation indicated the gain was prompt-heavy.
- Phase 3.5 controlled prompt-equal comparison: base+best_prompt 3/30 -> adapter_v3+best_prompt 5/30 (+6.7pp).

Do not use the unsupported old Phase 2 claim (23/30 -> 26/30) unless its artifact set is recovered.

## Issues Fixed During Verification

- Notebook initially ran from bare `/content`; fixed with repo clone + bundle overlay.
- Local Phase 3.5 files were not pushed to GitHub; fixed with `phase3_5_bundle.zip` overlay.
- Training failed on missing `trl`/`bitsandbytes`; fixed with dependency install cell.
- Adapter eval failed on incompatible `torchao==0.10.0`; fixed by upgrading to `torchao>=0.16.0`.

## Limitations

- This is a Colab T4 result, not an AMD result.
- Absolute score remains low (5/30), so this is a directional adapter-contribution pass, not a production-quality model.
- The adapter regressed some individual problems that base+prompt passed; future refinement should inspect per-problem flips.
- The SFT dataset is still small (61 examples).

## Conclusion

Phase 3.5 is **PASS**. The core objective was met: `adapter_v3 + best_prompt` outperformed `base + best_prompt` under a controlled same-prompt comparison.
