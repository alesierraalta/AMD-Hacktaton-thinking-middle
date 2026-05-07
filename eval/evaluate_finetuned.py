import argparse
import json
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
    parser.add_argument("--prompt_template", type=str, default=None,
                        help="Prompt template name (e.g., thinkanywhere_qwen_instruct)")
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
    print(f"Evaluating fine-tuned: base={args.base_model}, adapter={args.adapter_path}, problems={args.problems_path}, template={args.prompt_template}")

    metadata = _load_metadata_json(args.metadata_json)
    if metadata:
        print(f"Metadata injected: {list(metadata.keys())}")

    summary = evaluate_model(
        model_name=args.base_model,
        problems_path=args.problems_path,
        output_path=args.output_path,
        adapter_path=args.adapter_path,
        mock=args.mock,
        timeout=args.timeout,
        metadata=metadata,
        prompt_template=args.prompt_template,
    )

    print(f"Fine-tuned results saved to {args.output_path}")
    print(f"Summary: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.1f}%)")


if __name__ == "__main__":
    main()
