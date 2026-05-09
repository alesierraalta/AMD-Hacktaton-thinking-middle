import json
import os
import subprocess
import tempfile
import pytest

def test_generate_sft_v4_runs_and_outputs_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_v4.jsonl")
        
        # Run the script
        result = subprocess.run([
            "python", "scripts/generate_sft_v4.py",
            "--output", output_path,
            "--num_variations", "2",
            "--num_boundary", "2",
            "--num_diverse", "2"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert os.path.exists(output_path)
        
        with open(output_path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        
        # 2 targets * 2 variations + 2 boundary + 2 diverse = 8
        assert len(lines) == 8
        
        for line in lines:
            data = json.loads(line)
            assert "id" in data
            assert "prompt" in data
            assert "raw_output" in data
            
            # Check for structured reasoning flow
            raw_output = data["raw_output"]
            assert "<thinkanywhere>" in raw_output
            assert "</thinkanywhere>" in raw_output
            assert "Observation:" in raw_output
            assert "Plan:" in raw_output
            assert "Edge Cases:" in raw_output
            assert "Verification:" in raw_output
            
            # Check for python code block
            assert "```python" in raw_output
            assert "```" in raw_output.split("```python")[1]

def test_generate_sft_v4_contains_regression_targets():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_v4.jsonl")
        subprocess.run([
            "python", "scripts/generate_sft_v4.py",
            "--output", output_path,
            "--num_variations", "1",
            "--num_boundary", "0",
            "--num_diverse", "0"
        ])
        
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "frequency_map" in content
        assert "is_leap_year" in content
