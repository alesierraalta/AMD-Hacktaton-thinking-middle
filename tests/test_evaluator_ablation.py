"""
Tests for eval/evaluator.py ablation matrix and metadata injection.

Covers tasks:
  2.1  — --prompt_template and --language CLI flags (implicit in run_ablation)
  2.2  — 4-arm ablation dispatch
  2.3  — metadata block injection per record
  4.2  — metadata injection in mock mode
"""
import json
import os
import tempfile
from unittest import mock

import pytest


class TestMetadataInjection:
    """Task 2.3 / 4.2 — metadata block must appear in every JSONL record."""

    def test_metadata_present_when_metadata_arg_provided(self):
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.jsonl")
            meta = {
                "model_id": "Qwen/Qwen1.5-1.8B-Chat",
                "adapter_path": "none",
                "prompt_template_id": "thinkanywhere_qwen_instruct",
                "language": "en",
                "hardware": "Colab T4",
            }
            ev.evaluate_model(
                model_name="mock-model",
                problems_path="data/problems.jsonl",
                output_path=output_path,
                mock=True,
                metadata=meta,
            )
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    assert "metadata" in record
                    assert record["metadata"]["model_id"] == "Qwen/Qwen1.5-1.8B-Chat"
                    assert record["metadata"]["hardware"] == "Colab T4"

    def test_metadata_absent_when_metadata_arg_is_none(self):
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
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    assert "metadata" not in record

    def test_metadata_fields_match_ablation_arm_definition(self):
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.jsonl")
            arm_meta = {
                "model_id": "Qwen/Qwen1.5-1.8B-Chat",
                "adapter_path": "adapters/phase2",
                "prompt_template_id": "thinkanywhere_qwen_es",
                "language": "es",
                "hardware": "Colab T4",
            }
            ev.evaluate_model(
                model_name="mock-model",
                problems_path="data/problems.jsonl",
                output_path=output_path,
                mock=True,
                metadata=arm_meta,
            )
            with open(output_path, "r", encoding="utf-8") as f:
                first = next(line for line in f if line.strip())
            record = json.loads(first)
            m = record["metadata"]
            assert m["prompt_template_id"] == "thinkanywhere_qwen_es"
            assert m["language"] == "es"
            assert m["adapter_path"] == "adapters/phase2"


class TestAblationArms:
    """Task 2.2 — ABLATION_ARMS constant defines the 4-arm matrix."""

    def test_four_arms_defined(self):
        from eval.evaluator import ABLATION_ARMS
        assert len(ABLATION_ARMS) == 4

    def test_arm_labels_are_unique(self):
        from eval.evaluator import ABLATION_ARMS
        labels = [a["label"] for a in ABLATION_ARMS]
        assert len(labels) == len(set(labels))

    def test_arms_cover_base_and_finetuned(self):
        from eval.evaluator import ABLATION_ARMS
        arms_by_adapter = {a["arm"]: a["label"] for a in ABLATION_ARMS}
        # Arms C and D are the finetuned arms (have adapter in run_ablation)
        # Arms A and B are base model arms
        assert "A" in arms_by_adapter
        assert "B" in arms_by_adapter
        assert "C" in arms_by_adapter
        assert "D" in arms_by_adapter

    def test_spanish_arm_uses_en_language(self):
        from eval.evaluator import ABLATION_ARMS
        d_arm = next(a for a in ABLATION_ARMS if a["arm"] == "D")
        # Phase 3 correction: Arm D uses minimal_python_function (no Spanish)
        assert d_arm["language"] == "en"
        assert d_arm["prompt_template_id"] == "minimal_python_function"


class TestRunAblation:
    """Task 2.2 / 2.4 — run_ablation dispatches to 4 arms with cache clearing."""

    def test_run_ablation_writes_four_jsonl_files(self):
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            result = ev.run_ablation(
                problems_path="data/problems.jsonl",
                output_dir=tmpdir,
                base_model_name="Qwen/Qwen1.5-1.8B-Chat",
                mock=True,
            )

            for arm in ("A", "B", "C", "D"):
                path = os.path.join(tmpdir, f"arm_{arm}.jsonl")
                assert os.path.exists(path), f"arm_{arm}.jsonl not found"

    def test_run_ablation_returns_summary_per_arm(self):
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            result = ev.run_ablation(
                problems_path="data/problems.jsonl",
                output_dir=tmpdir,
                base_model_name="Qwen/Qwen1.5-1.8B-Chat",
                mock=True,
            )

            assert set(result.keys()) == {"A", "B", "C", "D"}
            for arm_result in result.values():
                assert "total" in arm_result
                assert "passed" in arm_result
                assert "pass_rate" in arm_result

    def test_run_ablation_arm_d_sets_minimal_python_template(self):
        """Arm D records must use minimal_python_function and language=en (Phase 3 correction)."""
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            ev.run_ablation(
                problems_path="data/problems.jsonl",
                output_dir=tmpdir,
                base_model_name="Qwen/Qwen1.5-1.8B-Chat",
                mock=True,
            )
            d_path = os.path.join(tmpdir, "arm_D.jsonl")
            with open(d_path, "r", encoding="utf-8") as f:
                first = next(line for line in f if line.strip())
            record = json.loads(first)
            assert record["metadata"]["language"] == "en"
            assert record["metadata"]["prompt_template_id"] == "minimal_python_function"

    def test_run_ablation_base_arms_have_no_adapter(self):
        """Arms A and B must carry adapter_path='none' (base model)."""
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            ev.run_ablation(
                problems_path="data/problems.jsonl",
                output_dir=tmpdir,
                base_model_name="Qwen/Qwen1.5-1.8B-Chat",
                mock=True,
            )
            for arm in ("A", "B"):
                path = os.path.join(tmpdir, f"arm_{arm}.jsonl")
                with open(path, "r", encoding="utf-8") as f:
                    first = next(line for line in f if line.strip())
                record = json.loads(first)
                assert record["metadata"]["adapter_path"] == "none"

    def test_run_ablation_finetuned_arms_have_adapter_path_set(self):
        """Arms C and D must carry the provided adapter_path."""
        import eval.evaluator as ev

        with tempfile.TemporaryDirectory() as tmpdir:
            ev.run_ablation(
                problems_path="data/problems.jsonl",
                output_dir=tmpdir,
                base_model_name="Qwen/Qwen1.5-1.8B-Chat",
                adapter_path="adapters/phase2",
                mock=True,
            )
            for arm in ("C", "D"):
                path = os.path.join(tmpdir, f"arm_{arm}.jsonl")
                with open(path, "r", encoding="utf-8") as f:
                    first = next(line for line in f if line.strip())
                record = json.loads(first)
                assert record["metadata"]["adapter_path"] == "adapters/phase2"


class TestClearModelCache:
    """Task 2.4 — _clear_model_cache must release GPU memory between arms."""

    def test_clear_model_cache_runs_without_error(self):
        from eval.evaluator import _clear_model_cache
        # Must not raise even when no GPU is present
        _clear_model_cache()  # should complete silently

    def test_clear_model_cache_idempotent(self):
        """Calling _clear_model_cache multiple times must not raise."""
        from eval.evaluator import _clear_model_cache
        _clear_model_cache()
        _clear_model_cache()
        _clear_model_cache()  # still silent after repeated calls
