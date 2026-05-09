import json
import pathlib
import pytest

NOTEBOOK_PATH = pathlib.Path("notebooks/codepause_phase_7_final_qwen25_7b.ipynb")

def test_notebook_exists():
    assert NOTEBOOK_PATH.exists()

def test_notebook_json_validity():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "cells" in data
    assert len(data["cells"]) > 0

def test_notebook_required_sections():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    sources = []
    for cell in data["cells"]:
        if cell["cell_type"] == "markdown":
            sources.append("".join(cell["source"]))
        elif cell["cell_type"] == "code":
            sources.append("".join(cell["source"]))
            
    content = "".join(sources)
    
    # Check for key phrases/sections
    assert "Setup & Environment Verification" in content
    assert "generate_sft_v7.py" in content
    assert "final_qwen25_7b.yaml" in content
    assert "run_phase7.py" in content
    assert "merge_phase7_adapter.py" in content
    assert "convert_phase7_to_gguf.sh" in content
    assert "Fallback (3B)" in content
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in content

def test_notebook_gpu_verification():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    code_cells = [cell for cell in data["cells"] if cell["cell_type"] == "code"]
    gpu_check_found = any("torch.cuda.get_device_name" in "".join(cell["source"]) for cell in code_cells)
    assert gpu_check_found, "GPU verification cell missing"
