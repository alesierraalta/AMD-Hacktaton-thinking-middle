import json
import os
import sys
import tempfile
from unittest import mock
import io

import pytest


class TestEvaluateFinetunedImport:
    def test_module_imports_without_error(self):
        import eval.evaluate_finetuned as ft

        assert hasattr(ft, "parse_args")
        assert hasattr(ft, "main")


class TestEvaluateFinetunedMock:
    def test_mock_produces_jsonl(self):
        import eval.evaluate_finetuned as ft

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "finetuned.jsonl")
            test_args = [
                "prog",
                "--base_model", "mock-base",
                "--adapter_path", "mock-adapter",
                "--problems_path", "data/problems.jsonl",
                "--output_path", output_path,
                "--mock",
            ]
            with mock.patch.object(sys, "argv", test_args):
                ft.main()
            assert os.path.exists(output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
            assert len(lines) > 0

    def test_mock_output_has_required_fields(self):
        import eval.evaluate_finetuned as ft

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "finetuned.jsonl")
            test_args = [
                "prog",
                "--base_model", "mock-base",
                "--adapter_path", "mock-adapter",
                "--problems_path", "data/problems.jsonl",
                "--output_path", output_path,
                "--mock",
            ]
            with mock.patch.object(sys, "argv", test_args):
                ft.main()
            with open(output_path, "r", encoding="utf-8") as f:
                first_line = json.loads(f.readline())
            for field in ["id", "prompt", "raw_output", "clean_code", "passed", "metrics"]:
                assert field in first_line, f"Missing field: {field}"

    def test_mock_output_metrics_has_six_fields(self):
        import eval.evaluate_finetuned as ft

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "finetuned.jsonl")
            test_args = [
                "prog",
                "--base_model", "mock-base",
                "--adapter_path", "mock-adapter",
                "--problems_path", "data/problems.jsonl",
                "--output_path", output_path,
                "--mock",
            ]
            with mock.patch.object(sys, "argv", test_args):
                ft.main()
            with open(output_path, "r", encoding="utf-8") as f:
                first_line = json.loads(f.readline())
            metrics = first_line["metrics"]
            assert len(metrics) == 6
            for field in [
                "tests_passed",
                "balanced_tags",
                "has_thinkanywhere",
                "thinkanywhere_blocks",
                "executable_after_strip",
                "clean_code_lines",
            ]:
                assert field in metrics

    def test_help_includes_all_options(self):
        import eval.evaluate_finetuned as ft

        test_args = ["prog", "--help"]
        with mock.patch.object(sys, "argv", test_args):
            import io

            with mock.patch("sys.stdout", new=io.StringIO()) as captured:
                try:
                    ft.parse_args()
                except SystemExit:
                    pass
        output = captured.getvalue()
        for opt in ["--base_model", "--adapter_path", "--problems_path", "--output_path", "--mock"]:
            assert opt in output, f"{opt} not in help"

    def test_calls_evaluator_with_adapter_path(self):
        import eval.evaluate_finetuned as ft

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "finetuned.jsonl")
            test_args = [
                "prog",
                "--base_model", "mock-base",
                "--adapter_path", "mock-adapter",
                "--problems_path", "data/problems.jsonl",
                "--output_path", output_path,
                "--mock",
            ]
            with mock.patch.object(sys, "argv", test_args):
                with mock.patch("eval.evaluate_finetuned.evaluate_model") as mock_eval:
                    mock_eval.return_value = {"total": 30, "passed": 15, "pass_rate": 50.0}
                    ft.main()
                    mock_eval.assert_called_once_with(
                        model_name="mock-base",
                        problems_path="data/problems.jsonl",
                        output_path=output_path,
                        adapter_path="mock-adapter",
                        mock=True,
                        timeout=5,
                        metadata={},
                    )


class TestEvaluateFinetunedMetadata:
    def test_metadata_json_flag_exists(self):
        import eval.evaluate_finetuned as ft

        test_args = ["prog", "--help"]
        with mock.patch.object(sys, "argv", test_args):
            with mock.patch("sys.stdout", new=io.StringIO()) as captured:
                try:
                    ft.parse_args()
                except SystemExit:
                    pass
        output = captured.getvalue()
        assert "--metadata_json" in output

    def test_metadata_passed_to_evaluator(self):
        import json
        import tempfile
        import eval.evaluate_finetuned as ft

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "finetuned.jsonl")
            meta_path = os.path.join(tmpdir, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"model_name": "test-base", "hardware": "Colab T4"}, f)

            test_args = [
                "prog",
                "--base_model", "mock-base",
                "--adapter_path", "mock-adapter",
                "--problems_path", "data/problems.jsonl",
                "--output_path", output_path,
                "--mock",
                "--metadata_json", meta_path,
            ]
            with mock.patch.object(sys, "argv", test_args):
                with mock.patch("eval.evaluate_finetuned.evaluate_model") as mock_eval:
                    mock_eval.return_value = {"total": 30, "passed": 15, "pass_rate": 50.0}
                    ft.main()
                    _, kwargs = mock_eval.call_args
                    assert kwargs["metadata"] == {"model_name": "test-base", "hardware": "Colab T4"}
