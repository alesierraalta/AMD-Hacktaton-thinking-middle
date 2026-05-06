import json
import os
import tempfile
from unittest import mock

import pytest


class TestEvaluatorImport:
    def test_module_imports_without_error(self):
        import eval.evaluator as ev

        assert hasattr(ev, "evaluate_model")


class TestEvaluatorMockMode:
    def test_mock_produces_jsonl_with_required_fields(self):
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.jsonl")
            summary = ev.evaluate_model(
                model_name="mock-model",
                problems_path="data/problems.jsonl",
                output_path=output_path,
                mock=True,
            )

            assert os.path.exists(output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
            assert len(lines) > 0

            first = json.loads(lines[0])
            for field in ["id", "prompt", "raw_output", "clean_code", "passed", "metrics"]:
                assert field in first

            assert "total" in summary
            assert "passed" in summary
            assert "pass_rate" in summary
            assert summary["total"] == len(lines)

    def test_mock_returns_summary_with_counts(self):
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.jsonl")
            summary = ev.evaluate_model(
                model_name="mock-model",
                problems_path="data/problems.jsonl",
                output_path=output_path,
                mock=True,
            )

            assert summary["total"] >= 30
            assert 0 <= summary["passed"] <= summary["total"]
            assert 0.0 <= summary["pass_rate"] <= 100.0

    def test_mock_exercises_strip_and_metrics(self):
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.jsonl")
            ev.evaluate_model(
                model_name="mock-model",
                problems_path="data/problems.jsonl",
                output_path=output_path,
                mock=True,
            )

            with open(output_path, "r", encoding="utf-8") as f:
                first = json.loads(f.readline())

            metrics = first["metrics"]
            assert "tests_passed" in metrics
            assert "balanced_tags" in metrics
            assert "has_thinkanywhere" in metrics
            assert "thinkanywhere_blocks" in metrics
            assert "executable_after_strip" in metrics
            assert "clean_code_lines" in metrics


class TestEvaluatorRealMode:
    def test_calls_load_model_and_generate(self):
        import eval.evaluator as ev

        mock_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.decode.return_value = "def add(a, b):\n    return a + b\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.jsonl")
            with mock.patch(
                "eval.evaluator.load_model_and_tokenizer",
                return_value=(mock_model, mock_tokenizer),
            ) as mock_load:
                summary = ev.evaluate_model(
                    model_name="real-model",
                    problems_path="data/problems.jsonl",
                    output_path=output_path,
                    mock=False,
                    max_new_tokens=128,
                    temperature=0.5,
                )

            mock_load.assert_called_once_with(
                "real-model", adapter_path=None, device_map="auto", torch_dtype="auto"
            )
            assert mock_model.generate.called
            _, kwargs = mock_model.generate.call_args
            assert kwargs.get("max_new_tokens") == 128
            assert kwargs.get("temperature") == 0.5

    def test_passes_adapter_path_to_loader(self):
        import eval.evaluator as ev

        mock_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.decode.return_value = "def add(a, b):\n    return a + b\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.jsonl")
            with mock.patch(
                "eval.evaluator.load_model_and_tokenizer",
                return_value=(mock_model, mock_tokenizer),
            ) as mock_load:
                ev.evaluate_model(
                    model_name="real-base",
                    problems_path="data/problems.jsonl",
                    output_path=output_path,
                    adapter_path="mock-adapter",
                    mock=False,
                )

            mock_load.assert_called_once_with(
                "real-base", adapter_path="mock-adapter", device_map="auto", torch_dtype="auto"
            )

    def test_mock_mode_does_not_load_model(self):
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.jsonl")
            with mock.patch(
                "eval.evaluator.load_model_and_tokenizer"
            ) as mock_load:
                ev.evaluate_model(
                    model_name="mock-model",
                    problems_path="data/problems.jsonl",
                    output_path=output_path,
                    mock=True,
                )
            mock_load.assert_not_called()

    def test_missing_problems_file_raises(self):
        import eval.evaluator as ev

        with pytest.raises(FileNotFoundError):
            ev.evaluate_model(
                model_name="mock-model",
                problems_path="nonexistent.jsonl",
                output_path="out.jsonl",
                mock=True,
            )

    def test_empty_problems_returns_zero_summary(self):
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            problems_path = os.path.join(tmpdir, "empty.jsonl")
            open(problems_path, "w").close()
            output_path = os.path.join(tmpdir, "results.jsonl")
            summary = ev.evaluate_model(
                model_name="mock-model",
                problems_path=problems_path,
                output_path=output_path,
                mock=True,
            )
            assert summary == {"total": 0, "passed": 0, "pass_rate": 0.0}

    def test_timeout_passed_to_run_code(self):
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.jsonl")
            with mock.patch("eval.evaluator.run_code") as mock_run:
                mock_run.return_value = {"passed": True}
                ev.evaluate_model(
                    model_name="mock-model",
                    problems_path="data/problems.jsonl",
                    output_path=output_path,
                    mock=True,
                    timeout=10,
                )
            for call in mock_run.call_args_list:
                assert call.kwargs.get("timeout") == 10

    def test_metadata_injected_into_records(self):
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.jsonl")
            meta = {"model_name": "test-smoke", "hardware": "Colab T4", "phase": "0.8"}
            ev.evaluate_model(
                model_name="mock-model",
                problems_path="data/problems.jsonl",
                output_path=output_path,
                mock=True,
                metadata=meta,
            )
            with open(output_path, "r", encoding="utf-8") as f:
                first = json.loads(f.readline())
            assert "metadata" in first
            assert first["metadata"]["model_name"] == "test-smoke"
            assert first["metadata"]["hardware"] == "Colab T4"

    def test_metadata_optional_omitted(self):
        """When metadata is None, no metadata key appears in records."""
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.jsonl")
            ev.evaluate_model(
                model_name="mock-model",
                problems_path="data/problems.jsonl",
                output_path=output_path,
                mock=True,
                metadata=None,
            )
            with open(output_path, "r", encoding="utf-8") as f:
                first = json.loads(f.readline())
            assert "metadata" not in first
