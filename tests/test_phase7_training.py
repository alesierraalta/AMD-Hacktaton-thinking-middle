import sys
import io
import pytest
from unittest import mock

def test_tokenizer_expansion_and_embedding_resize():
    """
    Test 2.1: Verify tokenizer adds 2 tokens and embeddings are resized.
    Also verify target_modules contains all 7 linear layers.
    """
    # Create mock modules for everything that might trigger the Windows cp1252 bug
    mock_trl = mock.MagicMock()
    mock_datasets = mock.MagicMock()
    mock_transformers = mock.MagicMock()
    mock_peft = mock.MagicMock()
    
    with mock.patch.dict("sys.modules", {
        "trl": mock_trl,
        "datasets": mock_datasets,
        "transformers": mock_transformers,
        "peft": mock_peft,
        "torch": mock.MagicMock(),
        "bitsandbytes": mock.MagicMock()
    }):
        import training.sft_lora as sft
        
        # Mock Tokenizer
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.add_special_tokens.return_value = 2  # Added 2 tokens
        mock_tokenizer.__len__.return_value = 1002 # original 1000 + 2
        mock_tokenizer.pad_token = "eos"
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Mock Model
        mock_model = mock.MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        
        # Mock Peft
        mock_peft_model = mock.MagicMock()
        mock_peft.get_peft_model.return_value = mock_peft_model
        
        test_args = [
            "prog",
            "--model_name", "mock-model",
            "--dataset_path", "tests/data/empty.jsonl",
            "--special_tokens",
            "--lora_all_linear",
        ]
        
        # Create dummy dataset
        import os
        os.makedirs("tests/data", exist_ok=True)
        with open("tests/data/empty.jsonl", "w") as f:
            f.write('{"text": "foo"}\n')
            
        with mock.patch.object(sys, "argv", test_args):
            sft.main()

        # Verification
        mock_tokenizer.add_special_tokens.assert_called_once()
        added_tokens_dict = mock_tokenizer.add_special_tokens.call_args[0][0]
        assert "additional_special_tokens" in added_tokens_dict
        assert "<thinkanywhere>" in added_tokens_dict["additional_special_tokens"]
        assert "</thinkanywhere>" in added_tokens_dict["additional_special_tokens"]
        
        mock_model.resize_token_embeddings.assert_called_once_with(1002)
        
        # Verify LoRA config used all 7 modules
        assert mock_peft.LoraConfig.called
        config_kwargs = mock_peft.LoraConfig.call_args[1]
        expected_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert set(config_kwargs["target_modules"]) == set(expected_modules)

def test_pad_token_setting():
    """Verify pad_token is set if missing."""
    mock_trl = mock.MagicMock()
    mock_datasets = mock.MagicMock()
    mock_transformers = mock.MagicMock()
    mock_peft = mock.MagicMock()
    
    with mock.patch.dict("sys.modules", {
        "trl": mock_trl,
        "datasets": mock_datasets,
        "transformers": mock_transformers,
        "peft": mock_peft,
        "torch": mock.MagicMock()
    }):
        import training.sft_lora as sft
        
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "eos"
        mock_tokenizer.add_special_tokens.return_value = 0
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        
        mock_model = mock.MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        
        test_args = [
            "prog",
            "--model_name", "mock-model",
            "--dataset_path", "tests/data/empty.jsonl",
            "--special_tokens",
        ]
        
        with mock.patch.object(sys, "argv", test_args):
            sft.main()
                
        # Verification
        assert mock_tokenizer.pad_token == "eos"
