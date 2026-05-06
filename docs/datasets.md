# Datasets

## Overview

Three JSONL datasets drive the pipeline:

| File | Count | Purpose |
|------|-------|---------|
| `data/problems.jsonl` | 30 | Programming problems with tests for evaluation |
| `data/thinkanywhere_sft.jsonl` | 30 | SFT training examples with `<thinkanywhere>` tags |
| `data/eval_cases.jsonl` | 5 | Held-out evaluation cases |

## Schema: `problems.jsonl`

```json
{
  "id": 0,
  "prompt": "Write a function add(a, b) that returns the sum of two numbers.",
  "entry_point": "add",
  "tests": [
    "assert add(1, 2) == 3",
    "assert add(-1, 1) == 0",
    "assert add(0, 0) == 0"
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int or str | Yes | Unique problem identifier |
| `prompt` | str | Yes | Natural language problem description |
| `entry_point` | str | Yes | Function name the tests expect |
| `tests` | list[str] | Yes | pytest-style assert statements |

**Validation:**
- Every problem must have at least one test.
- Tests are executed as Python `assert` statements concatenated after the generated code.

## Schema: `thinkanywhere_sft.jsonl`

```json
{
  "text": "<think>The task is to write a simple addition function.</think>\ndef add(a, b):\n    <thinkanywhere>Verify inputs are numeric.</thinkanywhere>\n    return a + b\n",
  "problem_id": 0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | str | Yes | Full prompt + response with Think-Anywhere tags |
| `problem_id` | int or str | Yes | Links to `problems.jsonl` entry |

**SFT Tag Rules:**

1. **Balanced tags:** Every `<think>` must have a matching `</think>`. Every `<thinkanywhere>` must have a matching `</thinkanywhere>`.
2. **Valid after strip:** After removing all `<think>...</think>` and `<thinkanywhere>...</thinkanywhere>` blocks, the remaining text must be syntactically valid Python.
3. **Self-contained code:** Each example's code must run independently in the sandbox. Do not reference functions defined in other examples.
4. **`<thinkanywhere>` placement:** Position near real decisions (conditionals, loops, edge cases, recursion) rather than trivial or repetitive locations.

**Validation performed by tests:**
- `test_sft_all_have_balanced_tags` — 100% must pass.
- `test_sft_all_strip_to_executable` — 100% must compile after strip.
- `test_sft_pass_rate_at_least_80_percent` — At least 80% must pass problem tests after strip.

## Schema: `eval_cases.jsonl`

Same schema as `problems.jsonl`. Used for held-out evaluation to detect overfitting.

## Sandbox Self-Contained Requirement

The sandbox runner (`src/sandbox_runner.py`) executes each generated code snippet in a temporary file via `subprocess.run`. The temporary file contains **only**:

1. The generated code (after stripping tags).
2. The problem's test asserts.

There is no shared context between executions. Therefore:

- Do not write SFT examples that call helper functions defined in other examples.
- Do not assume imports from previous executions.
- All required imports must be inside the generated code.

## Dataset Validation Commands

```bash
# Validate all datasets via tests
python -m pytest tests/test_pipeline.py -v

# Dry-run SFT dataset load
python training/sft_lora.py --model_name mock --dataset_path data/thinkanywhere_sft.jsonl --output_dir results/sft --dry-run
```
