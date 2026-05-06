import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report from evaluation results")
    parser.add_argument("--input_path", type=str, default="results/baseline.jsonl", help="Path to single results (JSONL)")
    parser.add_argument("--baseline", type=str, help="Path to baseline results (JSONL)")
    parser.add_argument("--finetuned", type=str, help="Path to fine-tuned results (JSONL)")
    parser.add_argument("--out", type=str, help="Output path for report (Markdown)")
    return parser.parse_args()


def load_results(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_report(results: list) -> str:
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    lines = [
        "# Baseline Evaluation Report",
        "",
        f"Total problems: {total}",
        f"Passed: {passed}",
        f"Pass rate: {passed / total * 100:.1f}%" if total else "Pass rate: N/A",
        "",
        "## Per-problem results",
        "",
    ]
    for r in results:
        mid = r.get("id", "unknown")
        status = "PASS" if r.get("passed", False) else "FAIL"
        lines.append(f"- {mid}: {status}")
    return "\n".join(lines)


def generate_comparison_report(baseline_results: list, finetuned_results: list) -> str:
    baseline_total = len(baseline_results)
    baseline_passed = sum(1 for r in baseline_results if r.get("passed", False))
    finetuned_total = len(finetuned_results)
    finetuned_passed = sum(1 for r in finetuned_results if r.get("passed", False))

    # Extract metadata if present (first record with metadata wins)
    def _meta_key(results, key, default="N/A"):
        for r in results:
            m = r.get("metadata", {})
            if key in m:
                return m[key]
        return default

    baseline_meta = baseline_results[0].get("metadata", {}) if baseline_results else {}
    finetuned_meta = finetuned_results[0].get("metadata", {}) if finetuned_results else {}

    # Header with metadata if available
    lines = [
        "# Evaluation Comparison Report",
        "",
        f"| Metric | Baseline | Fine-tuned |",
        f"|--------|----------|------------|",
        f"| Total | {baseline_total} | {finetuned_total} |",
        f"| Passed | {baseline_passed} | {finetuned_passed} |",
    ]
    if baseline_total:
        lines.append(f"| Pass rate | {baseline_passed / baseline_total * 100:.1f}%")
    else:
        lines.append("| Pass rate | N/A")
    if finetuned_total:
        lines[-1] += f" | {finetuned_passed / finetuned_total * 100:.1f}% |"
    else:
        lines[-1] += " | N/A |"

    # Metadata provenance block (only if metadata is present)
    if baseline_meta or finetuned_meta:
        lines.extend([
            "",
            "## Provenance",
            "",
        ])
        if baseline_meta:
            lines.append(f"- **Baseline model**: {baseline_meta.get('model_name', 'N/A')}")
            if baseline_meta.get("adapter_path"):
                lines.append(f"  - adapter: {baseline_meta['adapter_path']}")
            lines.append(f"  - hardware: {baseline_meta.get('hardware', 'N/A')}, phase: {baseline_meta.get('phase', 'N/A')}")
        if finetuned_meta:
            lines.append(f"- **Fine-tuned model**: {finetuned_meta.get('model_name', 'N/A')}")
            if finetuned_meta.get("adapter_path"):
                lines.append(f"  - adapter: {finetuned_meta['adapter_path']}")
            lines.append(f"  - hardware: {finetuned_meta.get('hardware', 'N/A')}, phase: {finetuned_meta.get('phase', 'N/A')}")

    lines.extend([
        "",
        "## Per-problem comparison",
        "",
    ])

    baseline_by_id = {r.get("id", "unknown"): r for r in baseline_results}
    finetuned_by_id = {r.get("id", "unknown"): r for r in finetuned_results}
    all_ids = sorted(set(baseline_by_id.keys()) | set(finetuned_by_id.keys()))

    for mid in all_ids:
        b_pass = baseline_by_id.get(mid, {}).get("passed", False)
        f_pass = finetuned_by_id.get(mid, {}).get("passed", False)
        b_status = "PASS" if b_pass else "FAIL"
        f_status = "PASS" if f_pass else "FAIL"
        lines.append(f"- {mid}: Baseline={b_status}, Fine-tuned={f_status}")

    return "\n".join(lines)


def main():
    args = parse_args()

    if args.baseline and args.finetuned:
        baseline_results = load_results(args.baseline)
        finetuned_results = load_results(args.finetuned)
        report = generate_comparison_report(baseline_results, finetuned_results)
    else:
        results = load_results(args.input_path)
        report = generate_report(results)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
