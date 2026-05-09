# README SUBMISSION - CodePause

## What is CodePause?
CodePause is a research-oriented prototype designed to implement and evaluate **Think-Anywhere blocks** in Large Language Models (LLMs) for code generation. It explores how allowing a model to "pause and think" at any token position during generation can improve reasoning and code correctness.

## The Problem
Standard code generation models often follow a linear path without explicit intermediate reasoning. While chain-of-thought (CoT) at the beginning of a response helps, complex logic often requires the model to re-evaluate its approach *while* writing the actual code (e.g., during complex loops, boundary conditions, or recursion).

## How Think-Anywhere Blocks Work
Unlike traditional `<think>` blocks that appear only at the start of a response, CodePause enables the use of `<thinkanywhere>` tags inserted directly inside the code block. These tags represent "internal reasoning pauses" where the model can verify logic, check indices, or plan the next lines without those thoughts being part of the final executable code.

## Stripping Blocks Before Execution
To maintain compatibility with standard compilers and interpreters, CodePause uses a dedicated stripping utility. Before any code is executed or tested:
1. All `<think>...</think>` blocks are removed.
2. All `<thinkanywhere>...</thinkanywhere>` blocks are removed from within the code.
3. The resulting "clean" code is then saved and executed in a sandbox environment.

## Evaluation Methodology
Evaluation is performed using a "held-out" dataset of programming problems with associated unit tests.
- **Baseline**: The base model (`Qwen1.5-1.8B-Chat`) with a zero-shot prompt optimized for thinking.
- **Adapter**: The same base model loaded with a LoRA/QLoRA adapter trained on the `Think-Anywhere` SFT dataset.
- **Metric**: Pass@1 rate on the held-out test suite.

## Measured Results
- **Baseline Score**: 1/30 (3.3%)
- **Adapter Score**: 3/30 (10.0%)
- **Improvement (Delta)**: +2/30 (+6.7pp)
*Note: While absolute performance is low (prototype stage), the adapter successfully demonstrates a positive delta over the optimized baseline.*

## Reproducibility in Colab T4
The project is designed to be reproducible on a single NVIDIA T4 GPU (standard Google Colab environment).
1. Open the provided notebook: `notebooks/codepause_phase_5_dataset_v5_recovery_streaming.ipynb`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Load the adapter from `results/phase6_release_candidate/` (see Loading Instructions in `MODEL_CARD.md`).
4. Run the evaluation cells to verify the results.

## Artifact Location
The final adapter weights and configuration files are located in:
`results/phase6_release_candidate/`

## Limitations & Disclaimers
- **Prototype Level**: This is an early-stage research prototype, not a production-robust system.
- **Environment**: This project was developed and validated on **NVIDIA T4** hardware. **No AMD MI300X results are claimed.**
- **Absolute Score**: The pass rate is low, reflecting the difficulty of the task and the limited scale of the prototype.
- **Phase 4 Failure**: A previous attempt (Phase 4) failed to converge due to dataset quality issues; the current candidate uses the Phase 5 recovery path.
