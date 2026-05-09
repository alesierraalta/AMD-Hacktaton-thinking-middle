# Phase 3 Generalization & Ablation Report

## Executive Summary
This report presents results for model generalization on 30 unseen (held-out) problems and a systematic ablation study of prompting vs. adapter influences.

## Held-out Results (Generalization)
- **Baseline**: 0/30 (0.0%)
- **Fine-tuned**: 0/30 (0.0%)
- **Delta**: 0 problems (0.0 pp)

## Ablation Results (4 Arms)
- **Arm A (Base + Baseline Prompt)**: 0.0%
- **Arm B (Base + ThinkAnywhere Prompt)**: 0.0%
- **Arm C (Adapter + ThinkAnywhere Prompt)**: 0.0%
- **Arm D (Adapter + Minimal Python)**: 0.0%

## Hardware & Environment
- Model: Qwen/Qwen1.5-1.8B-Chat
- Hardware: MOCK-T4
- Mock: True
- Notebook: codepause_phase_3_generalization_ablation.ipynb

## Notes
- Phase 2 v2 results for comparison: 24/30 -> 25/30 (+3.3pp).
- No retraining was performed in this phase.
