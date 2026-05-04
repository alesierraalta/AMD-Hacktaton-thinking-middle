def build_sft_prompt(problem: dict) -> str:
    instruction = problem.get("prompt", "")
    entry_point = problem.get("entry_point", "")
    return (
        f"### Instruction\n{instruction}\n\n"
        f"### Response\n"
        f"<think>Plan: implement {entry_point} carefully.</think>\n"
        f"```python\n"
        f"def {entry_point}():\n"
        f"    <thinkanywhere>Validate edge cases here.</thinkanywhere>\n"
        f"    pass\n"
        f"```"
    )


def build_baseline_prompt(problem: dict) -> str:
    instruction = problem.get("prompt", "")
    return f"### Instruction\n{instruction}\n\n"
