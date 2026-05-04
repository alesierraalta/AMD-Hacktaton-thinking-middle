import pytest
from src.metrics import compute_metrics


class TestComputeMetrics:
    def test_all_fields_present(self):
        result = compute_metrics(
            clean_code="x = 1\n",
            raw_output="<think>plan</think>x = 1\n",
            test_result={"passed": True},
        )
        assert "tests_passed" in result
        assert "balanced_tags" in result
        assert "has_thinkanywhere" in result
        assert "thinkanywhere_blocks" in result
        assert "executable_after_strip" in result
        assert "clean_code_lines" in result

    def test_happy_path(self):
        result = compute_metrics(
            clean_code="def add(a, b):\n    return a + b\n",
            raw_output="<think>plan</think><thinkanywhere>check</thinkanywhere>def add(a, b):\n    return a + b\n",
            test_result={"passed": True},
        )
        assert result["tests_passed"] is True
        assert result["balanced_tags"] is True
        assert result["has_thinkanywhere"] is True
        assert result["thinkanywhere_blocks"] == 1
        assert result["executable_after_strip"] is True
        assert result["clean_code_lines"] == 2

    def test_unbalanced_tags(self):
        result = compute_metrics(
            clean_code="x = 1\n",
            raw_output="<think>unbalanced\n",
            test_result={"passed": False},
        )
        assert result["balanced_tags"] is False
        assert result["tests_passed"] is False

    def test_invalid_code_syntax_error(self):
        result = compute_metrics(
            clean_code="def broken(:\n    pass\n",
            raw_output="<think>plan</think>def broken(:\n    pass\n",
            test_result={"passed": False},
        )
        assert result["executable_after_strip"] is False

    def test_no_thinkanywhere(self):
        result = compute_metrics(
            clean_code="x = 1\n",
            raw_output="<think>plan</think>x = 1\n",
            test_result={"passed": True},
        )
        assert result["has_thinkanywhere"] is False
        assert result["thinkanywhere_blocks"] == 0

    def test_empty_clean_code(self):
        result = compute_metrics(
            clean_code="",
            raw_output="<think></think>",
            test_result={"passed": True},
        )
        assert result["clean_code_lines"] == 0
        assert result["executable_after_strip"] is False
