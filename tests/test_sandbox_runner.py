import pytest
from src.sandbox_runner import run_code


class TestRunCode:
    def test_correct_code_passes(self):
        code = "def add(a, b):\n    return a + b\n"
        tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        result = run_code(code, tests)
        assert result["passed"] is True
        assert result["returncode"] == 0
        assert result["timeout"] is False

    def test_failing_code_fails(self):
        code = "def add(a, b):\n    return a - b\n"
        tests = ["assert add(1, 2) == 3"]
        result = run_code(code, tests)
        assert result["passed"] is False
        assert result["timeout"] is False

    def test_infinite_loop_times_out(self):
        code = "while True:\n    pass\n"
        tests = ["assert True"]
        result = run_code(code, tests, timeout=1)
        assert result["timeout"] is True
        assert result["passed"] is False

    def test_empty_tests_pass(self):
        code = "x = 1\n"
        tests = []
        result = run_code(code, tests)
        assert result["passed"] is True
        assert result["returncode"] == 0

    def test_syntax_error_fails(self):
        code = "def broken(:\n"
        tests = ["assert True"]
        result = run_code(code, tests)
        assert result["passed"] is False
        assert result["returncode"] != 0
