"""Tests for Phase 3.5 evaluator --prompt_template support."""
import argparse
import sys
from unittest import mock


class TestEvaluateBaselinePromptSupport:
    def test_prompt_template_arg_exists(self):
        from eval import evaluate_baseline
        # Test that --prompt_template is a valid argument by checking the parser
        # We use parse_known_args to avoid required-arg errors
        test_args = [
            "prog",
            "--model_name", "Qwen/Qwen1.5-1.8B-Chat",
            "--problems_path", "data/test.jsonl",
            "--output_path", "results/test.jsonl",
            "--prompt_template", "best_phase3_prompt",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = evaluate_baseline.parse_args()
        assert hasattr(args, "prompt_template")
        assert args.prompt_template == "best_phase3_prompt"

    def test_prompt_template_best_phase3_prompt_accepted(self):
        from eval import evaluate_baseline
        test_args = [
            "prog",
            "--model_name", "Qwen/Qwen1.5-1.8B-Chat",
            "--problems_path", "data/heldout_problems_30.jsonl",
            "--output_path", "results/test.jsonl",
            "--prompt_template", "best_phase3_prompt",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = evaluate_baseline.parse_args()
        assert args.prompt_template == "best_phase3_prompt"


class TestEvaluateFinetunedPromptSupport:
    def test_prompt_template_arg_exists(self):
        from eval import evaluate_finetuned
        test_args = [
            "prog",
            "--base_model", "Qwen/Qwen1.5-1.8B-Chat",
            "--adapter_path", "results/test_adapter",
            "--problems_path", "data/test.jsonl",
            "--output_path", "results/test.jsonl",
            "--prompt_template", "best_phase3_prompt",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = evaluate_finetuned.parse_args()
        assert args.prompt_template == "best_phase3_prompt"

    def test_prompt_template_best_phase3_prompt_accepted(self):
        from eval import evaluate_finetuned
        test_args = [
            "prog",
            "--base_model", "Qwen/Qwen1.5-1.8B-Chat",
            "--adapter_path", "results/test_adapter",
            "--problems_path", "data/heldout_problems_30.jsonl",
            "--output_path", "results/test.jsonl",
            "--prompt_template", "best_phase3_prompt",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = evaluate_finetuned.parse_args()
        assert args.prompt_template == "best_phase3_prompt"
