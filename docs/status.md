# Status

## Phase Closure

### Phase 0: Pre-GPU Preparation

- **Archive ID:** `sdd/codepause-phase-0/archive-report` #3219
- **Tasks:** 22/22 complete
- **Tests:** 71 passed
- **Coverage:** 89%
- **Critical warnings:** 0

**Deliverables:**
- Repo structure (`src/`, `eval/`, `training/`, `app/`, `data/`, `tests/`)
- 30 problems + 30 SFT examples + core utilities
- Baseline evaluation and reporting
- README and mock demo

### Phase 0.5: Adapter-Aware Evaluation Gap

- **Archive ID:** `sdd/codepause-phase-0-5-adapter-eval/archive-report` #3231
- **Tests:** 95 passed
- **Coverage:** 91%
- **Critical warnings:** 0

**Deliverables:**
- `src/model_loader.py` — unified base model + LoRA adapter loader
- `eval/evaluator.py` — shared evaluation core
- `eval/evaluate_finetuned.py` — adapter-aware evaluation CLI
- Mock mode verified for both baseline and fine-tuned paths

## Readiness State

| Gate | Status | Evidence |
|------|--------|----------|
| Phase 0 complete | Closed | Archive #3219 |
| Phase 0.5 complete | Closed | Archive #3231 |
| All tests pass | Passing | 95/95, 91% coverage |
| Mock baseline eval | Working | `results/baseline_mock.jsonl` (30 records) |
| Mock fine-tuned eval | Working | `results/finetuned_mock.jsonl` (30 records) |
| Report generation | Working | `results/mock_report.md` |
| GPU-ready tag | Pushed | `phase-1-gpu-ready` |
| Operational runbook | Written | `sdd/codepause-phase-1-gpu-runbook.md` |

## Next Step

**Execute Phase 1 on AMD Developer Cloud MI300X.**

Use the operational runbook: [`sdd/codepause-phase-1-gpu-runbook.md`](../sdd/codepause-phase-1-gpu-runbook.md)

Phase 1 must prove:

```text
baseline → SFT/LoRA → evaluation → report
```

on real AMD GPU hardware. GRPO remains Phase 2.

## Blockers

None. The project is ready for GPU execution.

## Documentation Map

| Document | Purpose |
|----------|---------|
| `README.md` | Project landing page |
| `docs/architecture.md` | Module boundaries and data flow |
| `docs/datasets.md` | Dataset schemas and validation |
| `docs/evaluation.md` | Evaluation pipeline and reports |
| `docs/training.md` | SFT/LoRA and training modes |
| `docs/operations.md` | Cost discipline and GPU rules |
| `docs/testing.md` | Test suite reference |
| `docs/status.md` | This file |
| `sdd/codepause-phase-1-gpu-runbook.md` | Authoritative operational runbook |
