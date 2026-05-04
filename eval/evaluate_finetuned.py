import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.evaluator import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuned evaluation for Think-Anywhere")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--problems_path", type=str, required=True, help="Path to problems dataset (JSONL)")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for results (JSONL)")
    parser.add_argument("--mock", action="store_true", help="Use mock generation instead of real model")
    parser.add_argument("--timeout", type=int, default=5, help="Sandbox timeout in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Evaluating fine-tuned: base={args.base_model}, adapter={args.adapter_path}, problems={args.problems_path}")

    summary = evaluate_model(
        model_name=args.base_model,
        problems_path=args.problems_path,
        output_path=args.output_path,
        adapter_path=args.adapter_path,
        mock=args.mock,
        timeout=args.timeout,
    )

    print(f"Fine-tuned results saved to {args.output_path}")
    print(f"Summary: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.1f}%)")


if __name__ == "__main__":
    main()
