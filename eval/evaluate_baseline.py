import argparse
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
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Evaluating baseline: model={args.model_name}, problems={args.problems_path}")

    summary = evaluate_model(
        model_name=args.model_name,
        problems_path=args.problems_path,
        output_path=args.output_path,
        mock=args.mock,
        timeout=args.timeout,
    )

    print(f"Baseline results saved to {args.output_path}")
    print(f"Summary: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.1f}%)")


if __name__ == "__main__":
    main()
