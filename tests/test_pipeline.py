import json
import pytest
from src.strip_thinking import strip_thinking_blocks, has_balanced_tags
from src.sandbox_runner import run_code
from src.metrics import compute_metrics


class TestPipeline:
    def test_full_flow_on_sample_problem(self):
        problem = {
            "id": 0,
            "prompt": "Write a function add(a, b) that returns the sum.",
            "entry_point": "add",
            "tests": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
        }
        raw_output = (
            "<think>Plan: return a + b</think>\n"
            "def add(a, b):\n"
            "    <thinkanywhere>Check edge cases</thinkanywhere>\n"
            "    return a + b\n"
        )
        clean_code = strip_thinking_blocks(raw_output)
        test_result = run_code(clean_code, problem["tests"])
        metrics = compute_metrics(clean_code, raw_output, test_result)

        assert test_result["passed"] is True
        assert metrics["tests_passed"] is True
        assert metrics["balanced_tags"] is True
        assert metrics["has_thinkanywhere"] is True
        assert metrics["thinkanywhere_blocks"] == 1
        assert metrics["executable_after_strip"] is True


class TestDatasetValidation:
    def test_problems_dataset_has_30_items(self):
        with open("data/problems.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) >= 30

    def test_problems_dataset_required_fields(self):
        with open("data/problems.jsonl") as f:
            problems = [json.loads(line) for line in f if line.strip()]
        for p in problems:
            assert "id" in p
            assert "prompt" in p
            assert "entry_point" in p
            assert "tests" in p
            assert isinstance(p["tests"], list)
            assert len(p["tests"]) >= 1

    def test_sft_dataset_has_30_examples(self):
        with open("data/thinkanywhere_sft.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) >= 30

    def test_sft_dataset_has_text_field(self):
        with open("data/thinkanywhere_sft.jsonl") as f:
            examples = [json.loads(line) for line in f if line.strip()]
        for ex in examples:
            assert "text" in ex

    def test_sft_all_have_balanced_tags(self):
        with open("data/thinkanywhere_sft.jsonl") as f:
            examples = [json.loads(line) for line in f if line.strip()]
        for ex in examples:
            assert has_balanced_tags(ex["text"]) is True

    def test_sft_all_strip_to_executable(self):
        with open("data/thinkanywhere_sft.jsonl") as f:
            examples = [json.loads(line) for line in f if line.strip()]
        for ex in examples:
            clean = strip_thinking_blocks(ex["text"])
            assert clean.strip()
            compile(clean, "<string>", "exec")

    def test_sft_pass_rate_at_least_80_percent(self):
        with open("data/problems.jsonl") as f:
            problems = [json.loads(line) for line in f if line.strip()]
        with open("data/thinkanywhere_sft.jsonl") as f:
            examples = [json.loads(line) for line in f if line.strip()]
        assert len(problems) == len(examples) == 30

        passed = 0
        for prob, ex in zip(problems, examples):
            clean = strip_thinking_blocks(ex["text"])
            result = run_code(clean, prob["tests"])
            if result["passed"]:
                passed += 1

        assert passed / len(problems) >= 0.80

    def test_eval_cases_exists(self):
        with open("data/eval_cases.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) >= 1
