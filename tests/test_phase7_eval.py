import pytest
import json
import os
from unittest import mock

# We'll mock the actual model loading and generation to keep tests fast and deterministic.

def test_tag_format_validator():
    """Requirement: Tag Format Compliance (paired, no nesting)"""
    from eval.run_phase7 import validate_tags
    
    # Valid
    assert validate_tags("<thinkanywhere> reasoning </thinkanywhere> code") == True
    assert validate_tags("code <thinkanywhere> reasoning </thinkanywhere>") == True
    
    # Invalid: Unpaired
    assert validate_tags("<thinkanywhere> reasoning code") == False
    assert validate_tags("reasoning </thinkanywhere> code") == False
    
    # Invalid: Nested
    assert validate_tags("<thinkanywhere> <thinkanywhere> nested </thinkanywhere> </thinkanywhere>") == False
    
    # Invalid: Multiple blocks (not explicitly forbidden but usually we expect one; spec says "no nested")
    # Actually, spec says "matching opening and closing tags, no nested tags, code outside the block"
    # If there are multiple, as long as they aren't nested, it might be okay, but typically we want one.
    # Let's stick to the strict "no nesting" and "paired" rule.

def test_pass_at_1_calculation():
    """Requirement: Pass@1 math and comparison"""
    from eval.run_phase7 import compute_pass_rate
    
    results = [
        {"passed": True}, {"passed": True}, {"passed": False}, {"passed": True}
    ]
    # 3/4 = 75%
    assert compute_pass_rate(results) == 75.0

def test_regression_gate():
    """Requirement: Regression Gate (no new failures on shared set)"""
    from eval.run_phase7 import check_regression
    
    baseline_results = {
        "prob1": True,
        "prob2": True,
        "prob3": False
    }
    
    # Case 1: No regression
    new_results = {
        "prob1": True,
        "prob2": True,
        "prob3": True # Improved!
    }
    assert check_regression(new_results, baseline_results) == True
    
    # Case 2: Regression
    new_results_regressed = {
        "prob1": False, # Regressed
        "prob2": True,
        "prob3": False
    }
    assert check_regression(new_results_regressed, baseline_results) == False

def test_export_readiness_logic():
    """Requirement: All gates must pass for export_ready=True"""
    from eval.run_phase7 import evaluate_gates
    
    # All pass
    report = evaluate_gates(
        probe_diff=True,
        pass_rate=16.0,
        baseline_pass_rate=10.0,
        tag_pass_rate=90.0,
        has_regression=False
    )
    assert report["export_ready"] == True
    
    # Fail: Pass@1 too low
    report = evaluate_gates(
        probe_diff=True,
        pass_rate=14.0, # Below 15% (10 + 5)
        baseline_pass_rate=10.0,
        tag_pass_rate=90.0,
        has_regression=False
    )
    assert report["export_ready"] == False
    
    # Fail: Tag rate too low
    report = evaluate_gates(
        probe_diff=True,
        pass_rate=20.0,
        baseline_pass_rate=10.0,
        tag_pass_rate=80.0, # Below 85%
        has_regression=False
    )
    assert report["export_ready"] == False
    
    # Fail: Regression detected
    report = evaluate_gates(
        probe_diff=True,
        pass_rate=20.0,
        baseline_pass_rate=10.0,
        tag_pass_rate=90.0,
        has_regression=True
    )
    assert report["export_ready"] == False

@mock.patch("eval.run_phase7.evaluate_model")
@mock.patch("eval.run_phase7.probe_adapter")
def test_run_phase7_gate_flow(mock_probe, mock_eval, tmp_path):
    """Integration-style test for the full flow of run_phase7.py"""
    from eval.run_phase7 import run_gate
    
    # Setup mocks
    mock_probe.return_value = True
    
    # Mock evaluation results
    results_list = [
        {"id": "prob1", "passed": True, "raw_output": "<thinkanywhere> reasoning </thinkanywhere> code"},
        {"id": "prob2", "passed": True, "raw_output": "<thinkanywhere> reasoning </thinkanywhere> code"},
        {"id": "prob3", "passed": False, "raw_output": "bad output"}
    ]
    
    def side_effect(model_name, problems_path, output_path, adapter_path=None, mock=False):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for r in results_list:
                f.write(json.dumps(r) + "\n")
        return {
            "passed": 2,
            "total": 3,
            "pass_rate": 66.6
        }
    
    mock_eval.side_effect = side_effect
    
    # Mock baseline
    baseline_path = tmp_path / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump({"prob1": True, "prob2": False, "prob3": False}, f)
        
    report_path = tmp_path / "report.json"
    
    run_gate(
        base_model="mock",
        adapter_path="mock_adapter",
        problems_path="mock_problems.jsonl",
        baseline_path=str(baseline_path),
        output_report=str(report_path),
        threshold_boost=5.0
    )
    
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        report = json.load(f)
        
    assert "export_ready" in report
    assert report["pass_rate"] == 66.6
    assert report["tag_pass_rate"] == pytest.approx(66.6, abs=0.1) # 2/3 have tags

