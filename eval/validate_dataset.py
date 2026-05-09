"""
Dataset validation CLI for CodePause SFT/LoRA training.
Validates JSONL datasets for schema compliance and statistics.
"""
import argparse
import json
import sys
import os
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate dataset JSONL for CodePause training"
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        type=str,
        help="Path to JSONL dataset file",
    )
    parser.add_argument(
        "--sft_path",
        type=str,
        default=None,
        help="Path to SFT JSONL dataset file (alias for dataset_path, used by Phase 3.5 acceptance commands)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional Markdown quality report output path",
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


def _read_jsonl_records(dataset_path: str) -> list[dict]:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            with open(dataset_path, "r", encoding=encoding) as f:
                return [json.loads(line) for line in f if line.strip()]
        except UnicodeDecodeError:
            if encoding == "latin-1":
                raise
            continue
    return []


def write_quality_report(dataset_path: str, schema: str, out_path: str) -> None:
    records = _read_jsonl_records(dataset_path)
    stats = compute_stats(records, schema)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    lines = [
        "# Dataset Quality Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Schema: `{schema}`",
        f"- Total examples: {stats['total']}",
    ]
    if schema == "sft":
        lines.extend([
            f"- Average text length: {stats['avg_text_len']:.1f} chars",
            f"- Text length range: {stats['min_text_len']}–{stats['max_text_len']} chars",
            f"- `<thinkanywhere>` tags: {stats['thinkanywhere_tags']} ({stats['thinkanywhere_ratio']:.1%})",
        ])
    else:
        lines.extend([
            f"- Total tests: {stats['total_tests']}",
            f"- Average tests per problem: {stats['avg_tests']:.1f}",
        ])
    lines.extend(["", "Validation: **PASSED**", ""])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def validate_schema_sft(record: dict, line_num: int) -> list[str]:
    """Validate a single SFT record. Returns list of error messages."""
    errors = []
    # Support legacy 'text' OR hardened 'instruction'+'response'
    has_text = "text" in record
    has_v2 = "instruction" in record and "response" in record
    
    if not (has_text or has_v2):
        errors.append(f"line {line_num}: missing required fields (must have 'text' OR 'instruction' and 'response')")
    
    if has_text:
        text = record.get("text", "")
        if not isinstance(text, str):
            errors.append(f"line {line_num}: 'text' must be string, got {type(text).__name__}")
        elif len(text) < 5:
            errors.append(f"line {line_num}: 'text' too short (< 5 chars)")
        if "problem_id" not in record:
            errors.append(f"line {line_num}: missing required field 'problem_id' for legacy schema")
            
    if has_v2:
        for field in ("instruction", "response"):
            val = record.get(field, "")
            if not isinstance(val, str):
                errors.append(f"line {line_num}: '{field}' must be string, got {type(val).__name__}")
            elif len(val) < 2:
                errors.append(f"line {line_num}: '{field}' too short")
                
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
    else:
        # Check if entry_point appears in at least one test
        entry_point = record.get("entry_point", "");
        if entry_point and not any(entry_point in str(t) for t in tests):
            errors.append(f"line {line_num}: entry_point '{entry_point}' not found in any test")
        
    return errors


def validate_record(record: dict, line_num: int, schema: str) -> list[str]:
    if schema == "sft":
        return validate_schema_sft(record, line_num)
    return validate_schema_problems(record, line_num)

def check_duplicates(records, schema):
    from collections import Counter
    errors = []
    ids = [r.get("id") for r in records if r.get("id")]
    counts = Counter(ids)
    for rid, count in counts.items():
        if count > 1:
            errors.append(f"Duplicate ID detected: {rid} appears {count} times")
    return errors

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
        "total_chars": sum(len(str(r.get("text", r.get("prompt", r.get("instruction", "") + r.get("response", ""))))) for r in records),
    }

    if schema == "sft":
        text_lengths = []
        for r in records:
            if "text" in r:
                text_lengths.append(len(str(r["text"])))
            elif "instruction" in r:
                text_lengths.append(len(str(r["instruction"])) + len(str(r["response"])))
        
        stats["avg_text_len"] = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        stats["min_text_len"] = min(text_lengths) if text_lengths else 0
        stats["max_text_len"] = max(text_lengths) if text_lengths else 0

        # Tag balance check
        thinkanywhere_count = sum(1 for r in records if "<thinkanywhere>" in str(r.get("text", "")) or "<thinkanywhere>" in str(r.get("response", "")))
        stats["thinkanywhere_tags"] = thinkanywhere_count
        stats["thinkanywhere_ratio"] = thinkanywhere_count / len(records) if records else 0
    else:
        tests_counts = [len(r.get("tests", [])) for r in records]
        stats["avg_tests"] = sum(tests_counts) / len(tests_counts) if tests_counts else 0
        stats["total_tests"] = sum(tests_counts)

    return stats


def check_overlap(records: list[dict], reference_path: str = "data/problems.jsonl") -> list[str]:
    """Check for overlap between current records and Phase 2 main dataset."""
    errors = []
    if not os.path.exists(reference_path):
        return errors
        
    with open(reference_path, "r", encoding="utf-8") as f:
        ref_records = [json.loads(line) for line in f if line.strip()]
        
    ref_ids = {r.get("id") for r in ref_records if r.get("id")}
    ref_prompts = {r.get("prompt") for r in ref_records if r.get("prompt")}
    
    for r in records:
        rid = r.get("id")
        prompt = r.get("prompt")
        if rid in ref_ids:
            errors.append(f"Overlap detected: ID '{rid}' already exists in Phase 2 main dataset")
        if prompt in ref_prompts:
            errors.append(f"Overlap detected: Substantial prompt already exists in Phase 2 main dataset")
            
    return errors


def validate_dataset(
    dataset_path: str,
    schema: str = "sft",
    min_examples: int = 1,
    max_examples: int | None = None,
    quiet: bool = False,
    check_phase2_overlap: bool = True,
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

    # Duplicate check
    errors.extend(check_duplicates(records, schema))
    
    # Overlap check (Phase 3 Hard-stop)
    if schema == "problems" and check_phase2_overlap:
        # Don't check overlap if validating the main dataset itself
        if "data/problems.jsonl" not in dataset_path.replace("\\", "/"):
            errors.extend(check_overlap(records))

    # cp1252 fix check for SFT schema
    if schema == "sft" and records:
        sample_texts = []
        for r in records[:5]:
            if "text" in r:
                sample_texts.append(r["text"])
            elif "response" in r:
                sample_texts.append(r["response"])
        
        for i, text in enumerate(sample_texts, start=1):
            if text and not check_cp1252_fixes(text):
                errors.append(f"line {i}: possible cp1252 artifact detected")

    # Stats
    stats = compute_stats(records, schema)

    # Size checks
    if schema == "problems":
        # Phase 3 Hard-stop: held-out set must be exactly 30 records
        if stats["total"] != 30:
            errors.append(f"Phase 3 Requirement: 'problems' dataset must have exactly 30 records (found {stats['total']})")
    else:
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
    dataset_path = args.sft_path or args.dataset_path
    if not dataset_path:
        raise SystemExit("dataset_path or --sft_path is required")
    validate_dataset(
        dataset_path,
        schema=args.schema,
        min_examples=args.min_examples,
        max_examples=args.max_examples,
        quiet=args.quiet,
    )
    if args.out:
        write_quality_report(dataset_path, args.schema, args.out)
        if not args.quiet:
            print(f"Report saved to {args.out}")


if __name__ == "__main__":
    main()
