import sys
from unittest import mock


class TestDemoMockImport:
    def test_module_imports_without_error(self):
        import app.demo_mock as demo

        assert hasattr(demo, "run_pipeline")
        assert hasattr(demo, "main")


class TestDemoMockPipeline:
    def test_run_pipeline_returns_all_components(self):
        import app.demo_mock as demo

        result = demo.run_pipeline("Write a function that returns 42.")
        assert "prompt" in result
        assert "raw_output" in result
        assert "clean_code" in result
        assert "test_result" in result
        assert "metrics" in result

    def test_run_pipeline_metrics_has_six_fields(self):
        import app.demo_mock as demo

        result = demo.run_pipeline("Write a function that returns 42.")
        metrics = result["metrics"]
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

    def test_run_pipeline_raw_output_has_tags(self):
        import app.demo_mock as demo

        result = demo.run_pipeline("Write a function that returns 42.")
        assert "<think>" in result["raw_output"]
        assert "<thinkanywhere>" in result["raw_output"]

    def test_run_pipeline_clean_code_has_no_tags(self):
        import app.demo_mock as demo

        result = demo.run_pipeline("Write a function that returns 42.")
        assert "<think>" not in result["clean_code"]
        assert "<thinkanywhere>" not in result["clean_code"]

    def test_main_runs_without_error(self):
        import app.demo_mock as demo

        test_args = ["prog"]
        with mock.patch.object(sys, "argv", test_args):
            with mock.patch("builtins.print"):
                demo.main()
