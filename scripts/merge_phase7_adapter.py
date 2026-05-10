import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_ADAPTER_PATH = "results/phase7_release_candidate"
DEFAULT_OUT_DIR = "results/phase7_merged_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the Phase 7 LoRA/QLoRA adapter into the Qwen base model."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Torch dtype used while loading the base model.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to transformers. Use 'cpu' for CPU-only merge.",
    )
    return parser.parse_args()


def resolve_dtype(dtype: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path)
    out_dir = Path(args.out_dir)

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter path not found: {adapter_path}. Expected Phase 7 adapter files there."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )

    # Force resize token embeddings to match adapter if there's a mismatch
    # (Happens with some Qwen base vs instruct versions)
    print("Checking for tokenizer/embedding mismatch...")
    adapter_config_path = adapter_path / "adapter_config.json"
    if adapter_config_path.exists():
        import json
        with open(adapter_config_path, "r") as f:
            config_data = json.load(f)
            # PEFT sometimes stores modules_to_save or we can check the weight mismatch 
            # error directly if we could catch it, but checking common config fields 
            # or just blindly resizing if we know the target shape (151667) is better.
            # In this case, we saw the error: shape in current model is 152064, 
            # but adapter has 151667.
            target_vocab_size = 151667
            if base.config.vocab_size != target_vocab_size:
                print(f"Resizing embeddings from {base.config.vocab_size} to {target_vocab_size}")
                base.resize_token_embeddings(target_vocab_size)

    print(f"Applying adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.merge_and_unload()

    print(f"Saving merged model to: {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    print(f"Merged model saved to {out_dir}")


if __name__ == "__main__":
    main()
