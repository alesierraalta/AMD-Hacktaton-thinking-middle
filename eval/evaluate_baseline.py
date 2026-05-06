import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.evaluator import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline evaluation for Think-Anywhere")
    parser.add_argument("--model_name", type=str, default="default", help="Model name or path")
    parser.add_argument("--problems_path", type=str, required=True, help="Path to problems dataset (JSONL)")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for baseline results (JSONL)")
    parser.add_argument("--mock", action="store_true", help="Use mock generation instead of real model")
    parser.add_argument("--timeout", type=int, default=5, help="Sandbox timeout in seconds")
    parser.add_argument("--metadata_json", type=str, default=None,
                        help="Path to JSON file with metadata to inject into each result record (optional)")
    return parser.parse_args()


def _load_metadata_json(path: str | None):
    """Load metadata dict from a JSON file, returning empty dict if None or missing."""
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    print(f"Evaluating baseline: model={args.model_name}, problems={args.problems_path}")

    metadata = _load_metadata_json(args.metadata_json)
    if metadata:
        print(f"Metadata injected: {list(metadata.keys())}")

    summary = evaluate_model(
        model_name=args.model_name,
        problems_path=args.problems_path,
        output_path=args.output_path,
        mock=args.mock,
        timeout=args.timeout,
        metadata=metadata,
    )

    print(f"Baseline results saved to {args.output_path}")
    print(f"Summary: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.1f}%)")


if __name__ == "__main__":
    main()
