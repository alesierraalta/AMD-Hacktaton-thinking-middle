# Phase 1C: Colab-Only QLoRA Pivot

## Goal
Pivot the CodePause project from AMD MI300X to Google Colab T4 as the official execution environment for QLoRA fine-tuning.

## Functional Requirements
1. **Colab-Only Official Target:** The repository explicitly states Google Colab T4 as the official target. No AMD results can be claimed.
2. **Notebook Hard Stops:** The Colab notebook (`codepause_phase_1c_colab_only_qlora.ipynb`) MUST fail fast if:
   - A T4 GPU is not detected.
   - Google Drive is not mounted.
   - Local tests fail prior to training.
3. **Data Preservation:** The notebook MUST copy all generated artifacts (adapters, metrics, reports) to a designated folder in Google Drive.
4. **Model Progression:** The notebook executes in sequence: Qwen 0.5B (smoke test) -> Qwen 1.5B (primary).
5. **Config Alignment:** `config/models.yaml` defines the following roles: Qwen 0.5B (smoke), Qwen 1.5B (primary), Qwen 3B (stretch), Granite 3B (comparison), Gemma 4 E2B (optional).

## Non-Functional Requirements
1. **Deprecation:** AMD MI300X is documented as "historical/optional". TPU support is explicitly out of scope.
2. **Compatibility Preservation:**
   - QLoRA flags (`--load_in_4bit`, etc.) from Phase 0.8 MUST be preserved in the training script.
   - TRL 1.x `processing_class` fix MUST be preserved.
   - Windows/cp1252 UTF-8 fixes MUST be preserved in file I/O operations.
