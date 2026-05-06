import json
import os
import sys
import tempfile
from unittest import mock

import pytest


class TestMakeReportImport:
    def test_module_imports_without_error(self):
        import eval.make_report as report

        assert hasattr(report, "parse_args")
        assert hasattr(report, "main")
        assert hasattr(report, "load_results")
        assert hasattr(report, "generate_report")
        assert hasattr(report, "generate_comparison_report")


class TestMakeReport:
    def test_load_results_reads_jsonl(self):
        import eval.make_report as report

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "baseline.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "p1", "passed": True}) + "\n")
                f.write(json.dumps({"id": "p2", "passed": False}) + "\n")
            results = report.load_results(path)
            assert len(results) == 2
            assert results[0]["id"] == "p1"

    def test_generate_report_includes_summary(self):
        import eval.make_report as report

        results = [
            {"id": "p1", "passed": True, "metrics": {"tests_passed": True, "balanced_tags": True}},
            {"id": "p2", "passed": False, "metrics": {"tests_passed": False, "balanced_tags": True}},
        ]
        text = report.generate_report(results)
        assert "2" in text
        assert "p1" in text
        assert "p2" in text

    def test_main_produces_report(self):
        import eval.make_report as report

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "baseline.jsonl")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "p1", "passed": True, "metrics": {}}) + "\n")
            test_args = [
                "prog",
                "--input_path", input_path,
            ]
            captured = []
            with mock.patch.object(sys, "argv", test_args):
                with mock.patch("builtins.print", side_effect=captured.append):
                    report.main()
            assert any("p1" in str(line) for line in captured)

    def test_load_results_missing_file_raises(self):
        import eval.make_report as report

        try:
            report.load_results("nonexistent.jsonl")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


class TestMakeReportComparison:
    def test_comparison_report_includes_both(self):
        import eval.make_report as report

        baseline = [
            {"id": "p1", "passed": True, "metrics": {}},
            {"id": "p2", "passed": False, "metrics": {}},
        ]
        finetuned = [
            {"id": "p1", "passed": True, "metrics": {}},
            {"id": "p2", "passed": True, "metrics": {}},
        ]
        text = report.generate_comparison_report(baseline, finetuned)
        assert "Baseline" in text
        assert "Fine-tuned" in text
        assert "p1" in text
        assert "p2" in text

    def test_main_writes_comparison_to_out(self):
        import eval.make_report as report

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = os.path.join(tmpdir, "baseline.jsonl")
            finetuned_path = os.path.join(tmpdir, "finetuned.jsonl")
            out_path = os.path.join(tmpdir, "report.md")

            with open(baseline_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "p1", "passed": True, "metrics": {}}) + "\n")
            with open(finetuned_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "p1", "passed": True, "metrics": {}}) + "\n")

            test_args = [
                "prog",
                "--baseline", baseline_path,
                "--finetuned", finetuned_path,
                "--out", out_path,
            ]
            with mock.patch.object(sys, "argv", test_args):
                report.main()

            assert os.path.exists(out_path)
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Baseline" in content
            assert "Fine-tuned" in content

    def test_help_includes_comparison_options(self):
        import eval.make_report as report

        test_args = ["prog", "--help"]
        with mock.patch.object(sys, "argv", test_args):
            import io

            with mock.patch("sys.stdout", new=io.StringIO()) as captured:
                try:
                    report.parse_args()
                except SystemExit:
                    pass
        output = captured.getvalue()
        for opt in ["--baseline", "--finetuned", "--out", "--input_path"]:
            assert opt in output, f"{opt} not in help"


class TestMakeReportMetadata:
    def test_comparison_report_surfaces_metadata(self):
        import eval.make_report as report

        baseline = [
            {"id": "p1", "passed": True, "metadata": {"model_name": "Qwen2.5-0.5B", "hardware": "Colab T4", "phase": "0.8"}},
            {"id": "p2", "passed": False, "metadata": {"model_name": "Qwen2.5-0.5B", "hardware": "Colab T4", "phase": "0.8"}},
        ]
        finetuned = [
            {"id": "p1", "passed": True, "metadata": {"model_name": "Qwen2.5-0.5B", "hardware": "Colab T4", "phase": "0.8", "adapter_path": "/drive/adapters/0.5B"}},
            {"id": "p2", "passed": True, "metadata": {"model_name": "Qwen2.5-0.5B", "hardware": "Colab T4", "phase": "0.8", "adapter_path": "/drive/adapters/0.5B"}},
        ]
        text = report.generate_comparison_report(baseline, finetuned)
        assert "Provenance" in text
        assert "Qwen2.5-0.5B" in text
        assert "Colab T4" in text
        assert "0.8" in text
        assert "/drive/adapters/0.5B" in text

    def test_comparison_report_without_metadata_still_works(self):
        """No metadata present — report should still show the comparison table."""
        import eval.make_report as report

        baseline = [{"id": "p1", "passed": True}]
        finetuned = [{"id": "p1", "passed": True}]
        text = report.generate_comparison_report(baseline, finetuned)
        assert "Evaluation Comparison Report" in text
        assert "p1" in text
        assert "Provenance" not in text  # no metadata → no provenance section
