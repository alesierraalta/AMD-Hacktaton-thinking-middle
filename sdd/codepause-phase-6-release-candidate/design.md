# Design: CodePause Phase 6 — Release Candidate Model

## Architecture: RC Packaging

### Artifact Structure
```text
results/phase6_release_candidate/
├── adapter_config.json
├── adapter_model.safetensors
├── metadata_model.json
├── load_test_report.md
├── inference_smoke_report.md
├── eval_report.md
├── README_MODEL.md
└── checksums.txt
```

### Recreation Recipe (Phase 5 Clone)
- **Base Model**: `Qwen/Qwen1.5-1.8B-Chat`
- **Dataset**: `data/thinkanywhere_sft_v5.jsonl`
- **Method**: QLoRA (4-bit bitsandbytes)
- **LoRA Config**:
  - Rank: 8
  - Alpha: 16
  - Target Modules: `["q_proj", "v_proj"]`
  - Dropout: 0.05
- **Training Args**:
  - Learning Rate: 1e-4
  - Epochs: 1
  - Max Steps: 150
  - Batch Size: 1
  - Grad Accumulation: 8

### Metadata Schema
The `metadata_model.json` will follow the structure requested in the launch prompt.

### Verification Logic
1.  **Load Test**: Uses `eval/adapter_probe.py` logic but for general loading.
2.  **Smoke Test**: Uses a simplified version of `eval/evaluate_finetuned.py` for 3 fixed problems.
3.  **Checksums**: Uses standard `hashlib` SHA256.

### Implementation Patterns
- Use `rtk` prefix for all commands.
- Use `colab-mcp` for recreation if needed.
- Use `results/phase6_release_candidate/` as the immutable output target.
