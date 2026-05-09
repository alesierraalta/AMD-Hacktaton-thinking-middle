import sys
import io
import pytest
from unittest import mock
import os

def test_smoke_run_with_recipe():
    """
    Test 2.3: Verify smoke run can accept a recipe file.
    """
    import training.sft_lora as sft
    
    # Create dummy recipe
    os.makedirs("config/recipes", exist_ok=True)
    recipe_path = "config/recipes/test_recipe.yaml"
    with open(recipe_path, "w") as f:
        f.write("learning_rate: 0.0001\n")
        f.write("lora_rank: 16\n")
        f.write("special_tokens: true\n")
        f.write("lora_all_linear: true\n")

    test_args = [
        "prog",
        "--model_name", "mock",
        "--dataset_path", "tests/data/empty.jsonl",
        "--recipe", recipe_path, # THIS FLAG DOES NOT EXIST YET (RED)
        "--smoke",
        "--max_steps", "2",
    ]
    
    captured = io.StringIO()
    with mock.patch.object(sys, "argv", test_args):
        with mock.patch("sys.stdout", new=captured):
            try:
                sft.main()
            except SystemExit as e:
                if e.code != 0:
                    pytest.fail(f"Smoke run failed with exit code {e.code}")

    output = captured.getvalue()
    assert "SMOKE" in output
    assert "lr=0.0001" in output
    assert "rank=16" in output
