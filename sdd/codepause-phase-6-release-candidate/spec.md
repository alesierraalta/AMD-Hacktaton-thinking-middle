# Specification: CodePause Phase 6 — Release Candidate Model

## Requirements

### Requirement: Artifact Existence
The Phase 6 release candidate folder MUST contain the following files:
- `adapter_config.json`
- `adapter_model.safetensors`
- `metadata_model.json`
- `load_test_report.md`
- `inference_smoke_report.md`
- `eval_report.md`
- `README_MODEL.md`
- `checksums.txt`

### Requirement: Metadata Accuracy
The `metadata_model.json` MUST contain:
- project: CodePause
- phase: 6
- artifact_type: LoRA/QLoRA adapter
- base_model: Qwen/Qwen1.5-1.8B-Chat
- adapter_source_phase: Phase 5
- training_environment: Google Colab T4
- best_known_result: baseline 1/30, adapter 3/30, delta +2/30.

### Requirement: Load Verification
The system MUST verify that the adapter can be loaded alongside the base model using the PEFT library.
- **Scenario**: Successful Load
  - GIVEN a base model `Qwen/Qwen1.5-1.8B-Chat`
  - WHEN calling `load_model_and_tokenizer` with the RC adapter path
  - THEN the model MUST load without errors
  - AND the model class MUST be an instance of `PeftModel`.

### Requirement: Inference Smoke Test
The model MUST generate valid-looking Python code for common primitives.
- **Scenario**: Primitive Generation
  - GIVEN the loaded RC model
  - WHEN prompted to implement `add(a, b)`, `is_even(n)`, and `reverse_string(s)`
  - THEN the output MUST contain a thinking block
  - AND the output MUST contain the requested function implementation.

### Requirement: Checksum Integrity
The system MUST provide SHA256 checksums for the model weights and config to ensure artifact integrity.
