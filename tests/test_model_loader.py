import pytest
from unittest import mock


class TestModelLoaderImport:
    def test_module_imports_without_error(self):
        import src.model_loader as ml

        assert hasattr(ml, "load_model_and_tokenizer")


class TestModelLoaderWithoutAdapter:
    def test_calls_from_pretrained_for_model_and_tokenizer(self):
        import src.model_loader as ml

        mock_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()

        with mock.patch.object(ml, "AutoModelForCausalLM") as mock_auto_model, \
             mock.patch.object(ml, "AutoTokenizer") as mock_auto_tokenizer:
            mock_auto_model.from_pretrained.return_value = mock_model
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

            model, tokenizer = ml.load_model_and_tokenizer("mock-model")

            assert model is mock_model
            assert tokenizer is mock_tokenizer
            mock_auto_model.from_pretrained.assert_called_once_with(
                "mock-model", device_map="auto", torch_dtype="auto",
                quantization_config=None,
            )
            mock_auto_tokenizer.from_pretrained.assert_called_once_with("mock-model")

    def test_passes_custom_device_map_and_dtype(self):
        import src.model_loader as ml

        with mock.patch.object(ml, "AutoModelForCausalLM") as mock_auto_model, \
             mock.patch.object(ml, "AutoTokenizer") as mock_auto_tokenizer:
            ml.load_model_and_tokenizer(
                "mock-model", device_map="cpu", torch_dtype="float16"
            )
            mock_auto_model.from_pretrained.assert_called_once_with(
                "mock-model", device_map="cpu", torch_dtype="float16",
                quantization_config=None,
            )


class TestModelLoaderWithAdapter:
    def test_applies_peft_adapter_when_adapter_path_provided(self):
        import src.model_loader as ml

        mock_base_model = mock.MagicMock()
        mock_peft_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()

        with mock.patch.object(ml, "AutoModelForCausalLM") as mock_auto_model, \
             mock.patch.object(ml, "AutoTokenizer") as mock_auto_tokenizer, \
             mock.patch.object(ml, "PeftModel") as mock_peft, \
             mock.patch.object(ml, "_PEFT_AVAILABLE", True):
            mock_auto_model.from_pretrained.return_value = mock_base_model
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            mock_peft.from_pretrained.return_value = mock_peft_model

            model, tokenizer = ml.load_model_and_tokenizer(
                "mock-base", adapter_path="mock-adapter"
            )

            assert model is mock_peft_model
            assert tokenizer is mock_tokenizer
            mock_peft.from_pretrained.assert_called_once_with(
                mock_base_model, "mock-adapter"
            )

    def test_raises_import_error_if_peft_missing_and_adapter_requested(self):
        import src.model_loader as ml

        with mock.patch.object(ml, "AutoModelForCausalLM") as mock_auto_model, \
             mock.patch.object(ml, "AutoTokenizer") as mock_auto_tokenizer, \
             mock.patch.object(ml, "_PEFT_AVAILABLE", False):
            with pytest.raises(ImportError):
                ml.load_model_and_tokenizer("mock-base", adapter_path="mock-adapter")

    def test_import_error_raised_if_transformers_not_available(self):
        """Lines 3-5, 16-19: transformers missing → ImportError with specific message."""
        import src.model_loader as ml

        # Mock both to None to simulate import failure
        with mock.patch.object(ml, "AutoModelForCausalLM", None), \
             mock.patch.object(ml, "AutoTokenizer", None):
            with pytest.raises(ImportError, match="transformers is required"):
                ml.load_model_and_tokenizer("mock-model")

    def test_peft_not_available_but_no_adapter_still_works(self):
        """Line 10-12: peft missing but no adapter → still loads base model successfully."""
        import src.model_loader as ml

        with mock.patch.object(ml, "AutoModelForCausalLM") as mock_auto_model, \
             mock.patch.object(ml, "AutoTokenizer") as mock_auto_tokenizer, \
             mock.patch.object(ml, "_PEFT_AVAILABLE", False):
            mock_auto_model.from_pretrained.return_value = mock.MagicMock()
            mock_auto_tokenizer.from_pretrained.return_value = mock.MagicMock()
            # Should NOT raise - peft is only needed when adapter_path is provided
            model, tokenizer = ml.load_model_and_tokenizer("mock-model")
            assert model is not None
