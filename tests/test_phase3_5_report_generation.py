import json

from eval.make_report import generate_comparison_report, load_results


def test_phase3_5_report_generation_from_mock_outputs(tmp_path):
    baseline_path = tmp_path / "phase3_5_base_best_prompt.jsonl"
    adapter_path = tmp_path / "phase3_5_adapter_v3_best_prompt.jsonl"

    baseline_rows = [
        {"id": 1, "passed": True},
        {"id": 2, "passed": False},
        {"id": 3, "passed": False},
    ]
    adapter_rows = [
        {"id": 1, "passed": True},
        {"id": 2, "passed": True},
        {"id": 3, "passed": False},
    ]

    baseline_path.write_text(
        "".join(json.dumps(row) + "\n" for row in baseline_rows),
        encoding="utf-8",
    )
    adapter_path.write_text(
        "".join(json.dumps(row) + "\n" for row in adapter_rows),
        encoding="utf-8",
    )

    report = generate_comparison_report(
        load_results(str(baseline_path)),
        load_results(str(adapter_path)),
    )

    assert "# Evaluation Comparison Report" in report
    assert "| Passed | 1 | 2 |" in report
    assert "Baseline=FAIL, Fine-tuned=PASS" in report
