# Phase 0.8 — Colab T4 QLoRA Adapter Probe

## Scope

Validate the 4-bit QLoRA training pipeline on Google Colab T4 before AMD MI300X GPU credit is approved.

**In scope:**
- `config/models.yaml` — T4-safe model presets with 4-bit quantization configs
- `notebooks/codepause_colab_t4_qlora_probe.ipynb` — Colab notebook with hard stops
- `docs/colab_t4_qlora_probe.md` — Operations guide, hard stops, troubleshooting
- `training/sft_lora.py` smoke override for 4-bit flags
- `eval/evaluate_*.py` metadata injection

**Out of scope:**
- AMD MI300X execution (Phase 1 territory)
- TPU / JAX stack
- Unsloth as hard dependency
- Full convergence or benchmark reporting

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| `--load_in_4bit` gated behind explicit flag | Zero overhead when unused; preserves CPU smoke mode |
| Smoke overrides 4-bit with warning | Keeps smoke convenient; avoids accidental 4-bit on CPU |
| `config/models.yaml` (not JSON) | YAML is the project convention for config files |
| `notebooks/codepause_colab_t4_qlora_probe.ipynb` | Explicit filename matches design |
| Drive persistence required | Colab disconnects destroy local state; outputs must survive session death |
| Metadata injection via `--metadata_json` | Backward-compatible; no schema change to existing eval JSONL |

## Artifacts

| Artifact | Purpose |
|----------|---------|
| `config/models.yaml` | T4-safe presets (`colab-t4-qwen-1.5b`, `colab-t4-tinyllama`) |
| `notebooks/codepause_colab_t4_qlora_probe.ipynb` | Executable Colab notebook |
| `docs/colab_t4_qlora_probe.md` | Hard stops, troubleshooting, pre-flight checklist |
| `tests/test_training_qlora.py` | QLoRA flag parsing tests (Phase 2) |

## Evidence (Pending)

| Evidence | Status |
|----------|--------|
| Colab notebook run on T4 | **Pending** — requires manual Colab execution |
| Adapter exported to Drive | **Pending** — dependent on notebook run |
| Evaluation results | **Pending** — dependent on adapter |
| Training logs captured | **Pending** — dependent on notebook run |

> **Note**: No MI300X execution is claimed in Phase 0.8. All evidence is Colab T4-only.

## Chained PR Strategy

| PR | Scope | Files |
|----|-------|-------|
| PR 1 (this PR) | Foundation + docs | `config/models.yaml`, `notebooks/`, `docs/`, `sdd/README.md`, `README.md` |
| PR 2 | Core code + tests | `training/sft_lora.py` smoke/4bit, `eval/*.py` metadata, `src/model_loader.py`, tests |

## Status

```
Phase 0.8 — Foundation (Slice 1 of 2)
├── [x] config/models.yaml
├── [x] notebooks/codepause_colab_t4_qlora_probe.ipynb
├── [x] docs/colab_t4_qlora_probe.md
├── [x] sdd/README.md
└── [ ] README.md Phase 0.8 section update
    └── Phase 0.8 row in status table
```

**Next**: Slice 2 — CLI changes, model loader, eval metadata, tests.