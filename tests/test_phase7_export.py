import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys
import os

# Mock dependencies that might not be in the environment or are too heavy
sys.modules["torch"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["transformers"] = MagicMock()

from scripts.merge_phase7_adapter import parse_args, main

def test_merge_args_defaults():
    """Verify Phase 7 default paths and model."""
    with patch("sys.argv", ["merge_phase7_adapter.py"]):
        args = parse_args()
        assert args.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert args.adapter_path == "results/phase7_release_candidate"
        assert args.out_dir == "results/phase7_merged_model"

def test_merge_script_validates_adapter_path(tmp_path):
    """Ensure it raises FileNotFoundError if adapter doesn't exist."""
    adapter_path = tmp_path / "non_existent_adapter"
    out_dir = tmp_path / "out"
    
    with patch("sys.argv", ["merge_phase7_adapter.py", "--adapter-path", str(adapter_path), "--out-dir", str(out_dir)]):
        with pytest.raises(FileNotFoundError, match="Adapter path not found"):
            main()

@patch("scripts.merge_phase7_adapter.AutoTokenizer")
@patch("scripts.merge_phase7_adapter.AutoModelForCausalLM")
@patch("scripts.merge_phase7_adapter.PeftModel")
def test_merge_script_saves_with_safe_serialization(mock_peft, mock_model_class, mock_tokenizer_class, tmp_path):
    """Verify merge logic and safe_serialization=True."""
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text("{}")
    
    out_dir = tmp_path / "out"
    
    mock_base_model = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_base_model
    
    mock_peft_model = MagicMock()
    mock_peft.from_pretrained.return_value = mock_peft_model
    
    mock_merged_model = MagicMock()
    mock_peft_model.merge_and_unload.return_value = mock_merged_model
    
    with patch("sys.argv", ["merge_phase7_adapter.py", "--adapter-path", str(adapter_path), "--out-dir", str(out_dir)]):
        main()
        
    # Check if save_pretrained was called with safe_serialization=True
    mock_merged_model.save_pretrained.assert_called_once()
    args, kwargs = mock_merged_model.save_pretrained.call_args
    assert args[0] == out_dir
    assert kwargs.get("safe_serialization") is True

def test_gguf_script_path_logic():
    """
    Check if the GGUF script uses Phase 7 paths by reading the script file.
    Since it's a bash script, we'll verify its content logic.
    """
    script_path = Path("scripts/convert_phase7_to_gguf.sh")
    if not script_path.exists():
        pytest.fail("scripts/convert_phase7_to_gguf.sh does not exist yet")
        
    content = script_path.read_text()
    assert "results/phase7_merged_model" in content
    assert "results/phase7-final-qwen25-coder-7b.gguf" in content
