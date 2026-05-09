# DEMO SCRIPT - CodePause (2 Minutes)

## Goal
Demonstrate the core capability of CodePause: a model that thinks while it codes and improves performance compared to its base version.

---

### 1. The Environment (30s)
*   **Action**: Open the Google Colab Notebook `notebooks/codepause_phase_5_dataset_v5_recovery_streaming.ipynb`.
*   **Narrative**: "We are running in the official CodePause environment: a Google Colab T4. This setup allows us to run quantized inference on our 1.8B parameter model."

### 2. The Artifact (20s)
*   **Action**: Show the `results/phase6_release_candidate/` folder contents (`adapter_config.json`, `adapter_model.bin`).
*   **Narrative**: "Here is our Phase 6 Release Candidate. This is a LoRA adapter specifically trained to use Think-Anywhere blocks. It was recovered and refined after a major failure in Phase 4."

### 3. Inference & Thinking Behavior (30s)
*   **Action**: Run a smoke test inference cell. Highlight the `<thinkanywhere>` tags in the raw output.
*   **Narrative**: "Notice the model's output. It doesn't just write the Python function. It pauses with `<thinkanywhere>` tags to verify its logic—for example, checking if an index is out of bounds—before continuing to write the code."

### 4. Stripping & Execution (20s)
*   **Action**: Show the `strip_thinking_blocks()` function output.
*   **Narrative**: "Of course, this 'thought' text isn't valid Python. Our pipeline automatically strips these blocks before execution. What remains is clean, executable code that has benefited from the internal reasoning process."

### 5. Results & Limitations (20s)
*   **Action**: Display the `results_summary.md` table.
*   **Narrative**: "In our final evaluation, the adapter achieved a 10% pass rate compared to 3.3% for the baseline. While still a prototype with clear room for growth, it proves that the model is successfully using these 'thinking pauses' to improve its results."

---

## Technical Honesty Note
*If running live is not possible due to resource constraints:* "The notebook contains pre-computed results from our Phase 6 verification run. These match the reports in our artifact manifest exactly."
