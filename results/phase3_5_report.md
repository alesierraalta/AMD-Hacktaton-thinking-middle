# Evaluation Comparison Report

| Metric | Baseline | Fine-tuned |
|--------|----------|------------|
| Total | 30 | 30 |
| Passed | 3 | 5 |
| Pass rate | 10.0% | 16.7% |

## Delta

- Baseline score: 3/30 (10.0%)
- Fine-tuned score: 5/30 (16.7%)
- Delta: +2/30 (+6.7pp)
- Adapter contributes beyond prompt: YES

## Phase 3.5 Hard Gate

Requirement: `adapter_v3 + best_prompt > base + best_prompt`

Verdict: PASS

## Comparison Against Phase 3 Held-Out Results

- Phase 3 real T4 held-out: baseline 4/30 -> fine-tuned 8/30 (+13.3pp).
- Phase 3.5 same-prompt comparison: 3/30 -> 5/30 (+6.7pp).
- Valid Phase 2 caveat: use Phase 2 v2 real result 24/30 -> 25/30 (+3.3pp); do not use unsupported 23/30 -> 26/30 without recovered artifacts.

## Failure Taxonomy Summary

- Remaining failures are executable-code correctness failures, not just tag-placement failures.
- Persistent risk areas: boundary cases, parsing/indexing, empty inputs, loops, and recursion.
- Adapter regressions must be inspected before scaling.

## Per-problem comparison

- 30: Baseline=FAIL, Fine-tuned=FAIL
- 31: Baseline=FAIL, Fine-tuned=FAIL
- 32: Baseline=FAIL, Fine-tuned=FAIL
- 33: Baseline=FAIL, Fine-tuned=PASS
- 34: Baseline=FAIL, Fine-tuned=PASS
- 35: Baseline=FAIL, Fine-tuned=FAIL
- 36: Baseline=FAIL, Fine-tuned=FAIL
- 37: Baseline=FAIL, Fine-tuned=FAIL
- 38: Baseline=FAIL, Fine-tuned=FAIL
- 40: Baseline=FAIL, Fine-tuned=FAIL
- 41: Baseline=PASS, Fine-tuned=FAIL
- 42: Baseline=PASS, Fine-tuned=PASS
- 43: Baseline=FAIL, Fine-tuned=FAIL
- 44: Baseline=FAIL, Fine-tuned=FAIL
- 45: Baseline=PASS, Fine-tuned=FAIL
- 46: Baseline=FAIL, Fine-tuned=FAIL
- 47: Baseline=FAIL, Fine-tuned=FAIL
- 48: Baseline=FAIL, Fine-tuned=FAIL
- 49: Baseline=FAIL, Fine-tuned=FAIL
- 50: Baseline=FAIL, Fine-tuned=FAIL
- 51: Baseline=FAIL, Fine-tuned=FAIL
- 52: Baseline=FAIL, Fine-tuned=FAIL
- 53: Baseline=FAIL, Fine-tuned=FAIL
- 54: Baseline=FAIL, Fine-tuned=FAIL
- 55: Baseline=FAIL, Fine-tuned=FAIL
- 56: Baseline=FAIL, Fine-tuned=PASS
- 57: Baseline=FAIL, Fine-tuned=FAIL
- 58: Baseline=FAIL, Fine-tuned=FAIL
- 59: Baseline=FAIL, Fine-tuned=PASS
- 60: Baseline=FAIL, Fine-tuned=FAIL

## Aggregate Flips

- Adapter gains: 4 (33, 34, 56, 59)
- Adapter regressions: 2 (41, 45)

## Limitations

- This result should not be claimed as an AMD result unless run on AMD hardware.
- Absolute pass rate may still be low even when the hard gate passes.
- A positive adapter delta does not remove the need for failure-driven dataset refinement.

## Next Recommended Step

Refine the dataset around persistent failures and adapter regressions before scaling the model.