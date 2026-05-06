import json
import os
import sys
import tempfile
from unittest import mock
import io


class TestEvaluateBaselineImport:
    def test_module_imports_without_error(self):
        import eval.evaluate_baseline as baseline

        assert hasattr(baseline, "parse_args")
        assert hasattr(baseline, "main")


class TestEvaluateBaselineMock:
    def test_mock_produces_jsonl(self):
        import eval.evaluate_baseline as baseline

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "baseline.jsonl")
            test_args = [
                "prog",
                "--model_name", "mock",
                "--problems_path", "data/problems.jsonl",
                "--output_path", output_path,
                "--mock",
            ]
            with mock.patch.object(sys, "argv", test_args):
                baseline.main()
            assert os.path.exists(output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
            assert len(lines) > 0

    def test_mock_output_has_required_fields(self):
        import eval.evaluate_baseline as baseline

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "baseline.jsonl")
            test_args = [
                "prog",
                "--model_name", "mock",
                "--problems_path", "data/problems.jsonl",
                "--output_path", output_path,
                "--mock",
            ]
            with mock.patch.object(sys, "argv", test_args):
                baseline.main()
            with open(output_path, "r", encoding="utf-8") as f:
                first_line = json.loads(f.readline())
            for field in ["id", "prompt", "raw_output", "clean_code", "passed", "metrics"]:
                assert field in first_line, f"Missing field: {field}"

    def test_mock_output_metrics_has_six_fields(self):
        import eval.evaluate_baseline as baseline

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "baseline.jsonl")
            test_args = [
                "prog",
                "--model_name", "mock",
                "--problems_path", "data/problems.jsonl",
                "--output_path", output_path,
                "--mock",
            ]
            with mock.patch.object(sys, "argv", test_args):
                baseline.main()
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
        import eval.evaluate_baseline as baseline

        test_args = ["prog", "--help"]
        with mock.patch.object(sys, "argv", test_args):
            import io

            with mock.patch("sys.stdout", new=io.StringIO()) as captured:
                try:
                    baseline.parse_args()
                except SystemExit:
                    pass
        output = captured.getvalue()
        for opt in ["--model_name", "--problems_path", "--output_path", "--mock"]:
            assert opt in output, f"{opt} not in help"


class TestEvaluateBaselineMetadata:
    def test_metadata_json_flag_exists(self):
        import eval.evaluate_baseline as baseline

        test_args = ["prog", "--help"]
        with mock.patch.object(sys, "argv", test_args):
            with mock.patch("sys.stdout", new=io.StringIO()) as captured:
                try:
                    baseline.parse_args()
                except SystemExit:
                    pass
        output = captured.getvalue()
        assert "--metadata_json" in output

    def test_metadata_injected_into_jsonl(self):
        import json
        import tempfile
        import eval.evaluate_baseline as baseline

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "baseline.jsonl")
            meta_path = os.path.join(tmpdir, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"model_name": "test-model", "hardware": "Colab T4", "phase": "0.8"}, f)

            test_args = [
                "prog",
                "--model_name", "mock",
                "--problems_path", "data/problems.jsonl",
                "--output_path", output_path,
                "--mock",
                "--metadata_json", meta_path,
            ]
            with mock.patch.object(sys, "argv", test_args):
                baseline.main()

            with open(output_path, "r", encoding="utf-8") as f:
                first_line = json.loads(f.readline())
            assert "metadata" in first_line
            assert first_line["metadata"]["model_name"] == "test-model"
            assert first_line["metadata"]["hardware"] == "Colab T4"

    def test_missing_metadata_file_raises(self):
        import eval.evaluate_baseline as baseline

        test_args = [
            "prog",
            "--model_name", "mock",
            "--problems_path", "data/problems.jsonl",
            "--output_path", "/tmp/out.jsonl",
            "--mock",
            "--metadata_json", "nonexistent_meta.json",
        ]
        with mock.patch.object(sys, "argv", test_args):
            try:
                baseline.main()
                assert False, "Expected FileNotFoundError"
            except FileNotFoundError:
                pass
