"""
Dataset validation CLI for CodePause SFT/LoRA training.
Validates JSONL datasets for schema compliance and statistics.
"""
import argparse
import json
import sys
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate dataset JSONL for CodePause training"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to JSONL dataset file",
    )
    parser.add_argument(
        "--schema",
        choices=["sft", "problems"],
        default="sft",
        help="Dataset schema type",
    )
    parser.add_argument(
        "--min_examples",
        type=int,
        default=1,
        help="Minimum number of examples required",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples allowed (None = no limit)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors",
    )
    return parser.parse_args()


def validate_schema_sft(record: dict, line_num: int) -> list[str]:
    """Validate a single SFT record. Returns list of error messages."""
    errors = []
    if "text" not in record:
        errors.append(f"line {line_num}: missing required field 'text'")
    if "problem_id" not in record:
        errors.append(f"line {line_num}: missing required field 'problem_id'")
    text = record.get("text", "")
    if text and not isinstance(text, str):
        errors.append(f"line {line_num}: 'text' must be string, got {type(text).__name__}")
    if isinstance(text, str) and len(text) < 5:
        errors.append(f"line {line_num}: 'text' too short (< 5 chars)")
    return errors


def validate_schema_problems(record: dict, line_num: int) -> list[str]:
    """Validate a single problems record. Returns list of error messages."""
    errors = []
    for field in ("id", "prompt", "entry_point", "tests"):
        if field not in record:
            errors.append(f"line {line_num}: missing required field '{field}'")
    prompt = record.get("prompt", "")
    if prompt and not isinstance(prompt, str):
        errors.append(f"line {line_num}: 'prompt' must be string, got {type(prompt).__name__}")
    tests = record.get("tests", [])
    if not isinstance(tests, list):
        errors.append(f"line {line_num}: 'tests' must be list, got {type(tests).__name__}")
    elif len(tests) == 0:
        errors.append(f"line {line_num}: 'tests' list is empty")
    return errors


def validate_record(record: dict, line_num: int, schema: str) -> list[str]:
    if schema == "sft":
        return validate_schema_sft(record, line_num)
    return validate_schema_problems(record, line_num)


def check_cp1252_fixes(text: str) -> bool:
    """
    Check if text contains Windows cp1252 fix indicators.
    Returns True if the text appears to have been sanitized of common cp1252 artifacts.
    cp1252 bytes that commonly appear as decode errors: 0x93, 0x94, 0x96, 0x97
    are replaced with their Unicode equivalents.
    """
    # Check for common cp1252 smart-quote artifacts that weren't fixed
    cp1252_bad = ["\x93", "\x94", "\x96", "\x97"]
    for char in cp1252_bad:
        if char in text:
            return False
    return True


def compute_stats(records: list[dict], schema: str) -> dict:
    """Compute statistics over the dataset."""
    stats = {
        "total": len(records),
        "total_chars": sum(len(r.get("text", r.get("prompt", ""))) for r in records),
    }

    if schema == "sft":
        text_lengths = [len(r.get("text", "")) for r in records]
        stats["avg_text_len"] = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        stats["min_text_len"] = min(text_lengths) if text_lengths else 0
        stats["max_text_len"] = max(text_lengths) if text_lengths else 0

        # Tag balance check
        thinkanywhere_count = sum(1 for r in records if "<thinkanywhere>" in r.get("text", ""))
        stats["thinkanywhere_tags"] = thinkanywhere_count
        stats["thinkanywhere_ratio"] = thinkanywhere_count / len(records) if records else 0
    else:
        tests_counts = [len(r.get("tests", [])) for r in records]
        stats["avg_tests"] = sum(tests_counts) / len(tests_counts) if tests_counts else 0
        stats["total_tests"] = sum(tests_counts)

    return stats


def validate_dataset(
    dataset_path: str,
    schema: str = "sft",
    min_examples: int = 1,
    max_examples: int | None = None,
    quiet: bool = False,
) -> bool:
    """
    Validate a JSONL dataset file.

    Returns True if valid, raises SystemExit with error messages if invalid.
    """
    # Try UTF-8 first, fall back to cp1252 on Windows
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            with open(dataset_path, "r", encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            if encoding == "latin-1":
                raise
            continue

    records = []
    errors = []

    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            records.append(record)
        except json.JSONDecodeError as e:
            errors.append(f"line {i}: JSON parse error — {e}")

    # Schema validation
    for i, record in enumerate(records, start=1):
        errors.extend(validate_record(record, i, schema))

    # cp1252 fix check for SFT schema
    if schema == "sft" and records:
        sample_texts = [r.get("text", "") for r in records[:5]]
        for i, text in enumerate(sample_texts, start=1):
            if text and not check_cp1252_fixes(text):
                errors.append(f"line {i}: possible cp1252 artifact detected in 'text'")

    # Stats
    stats = compute_stats(records, schema)

    # Size checks
    if stats["total"] < min_examples:
        errors.append(f"dataset has {stats['total']} examples, minimum required is {min_examples}")
    if max_examples is not None and stats["total"] > max_examples:
        errors.append(f"dataset has {stats['total']} examples, maximum allowed is {max_examples}")

    # Report
    if not quiet:
        print(f"Dataset: {dataset_path}")
        print(f"Schema: {schema}")
        print(f"Total examples: {stats['total']}")

        if schema == "sft":
            print(f"Avg text length: {stats['avg_text_len']:.1f} chars")
            print(f"Text length range: {stats['min_text_len']}–{stats['max_text_len']} chars")
            print(f"<thinkanywhere> tags: {stats['thinkanywhere_tags']} ({stats['thinkanywhere_ratio']:.1%})")
        else:
            print(f"Total tests: {stats['total_tests']}")
            print(f"Avg tests per problem: {stats['avg_tests']:.1f}")

    if errors:
        print("\nERRORS:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        raise SystemExit(1)

    if not quiet:
        print("Validation: PASSED")

    return True


def main():
    args = parse_args()
    validate_dataset(
        args.dataset_path,
        schema=args.schema,
        min_examples=args.min_examples,
        max_examples=args.max_examples,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
