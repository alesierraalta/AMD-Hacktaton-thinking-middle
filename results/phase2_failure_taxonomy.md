# Phase 2 Failure Taxonomy

## Analysis of Phase 1C Evaluation Failures
Upon inspection of the 30 failures from the baseline evaluation (`baseline.jsonl` and `finetuned_mock.jsonl`), the root cause of the 0/30 score has been identified.

## Taxonomy
**1. Evaluator Extraction Contamination (100% of failures, 30/30)**
The evaluation framework fails to properly extract executable Python logic from the model's generation.
- **Syntax Errors from Prompt Headers**: The models output the initial `### Instruction` and prompt boilerplate inside the evaluated code blocks (`clean_code`). This results in `SyntaxError` when executed.
- **Missing Fenced Block Handling**: The extraction logic falls back to treating the entire output string as code, polluting the namespace with Markdown.
- **Thinking Block Bleed**: `thinkanywhere` or `think` tags are not always stripped out of the extracted executable path.

## Resolution Plan
Implement `extract_code()` with a strict fallback chain (prefer ```python, fallback any ```, fallback raw output), strip thinking blocks from the extracted portion, and validate with `ast.parse`.