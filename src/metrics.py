from src.strip_thinking import has_balanced_tags, count_thinkanywhere_blocks


def compute_metrics(clean_code: str, raw_output: str, test_result: dict) -> dict:
    balanced = has_balanced_tags(raw_output)
    thinkanywhere_count = count_thinkanywhere_blocks(raw_output)

    try:
        compile(clean_code, "<string>", "exec")
        executable = len(clean_code.strip()) > 0
    except SyntaxError:
        executable = False

    return {
        "tests_passed": test_result.get("passed", False),
        "balanced_tags": balanced,
        "has_thinkanywhere": thinkanywhere_count > 0,
        "thinkanywhere_blocks": thinkanywhere_count,
        "executable_after_strip": executable,
        "clean_code_lines": len(clean_code.splitlines()),
    }
