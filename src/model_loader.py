try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from peft import PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    PeftModel = None
    _PEFT_AVAILABLE = False


def load_model_and_tokenizer(model_name, adapter_path=None, device_map="auto", torch_dtype="auto", quantization_config=None):
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError(
            "transformers is required for load_model_and_tokenizer. "
            "Install it (e.g., pip install transformers)."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens so we can resize embeddings to match the adapter
    SPECIAL_TOKENS = ["<thinkanywhere>", "</thinkanywhere>"]
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        offload_folder="offload",
    )
    
    # Resize embeddings to match the adapter's vocabulary size
    model.resize_token_embeddings(len(tokenizer))

    if adapter_path is not None:
        if not _PEFT_AVAILABLE or PeftModel is None:
            raise ImportError(
                "peft is required to load an adapter. "
                "Install it (e.g., pip install peft)."
            )
        model = PeftModel.from_pretrained(model, adapter_path, offload_folder="offload")

    return model, tokenizer
