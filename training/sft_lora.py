import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="SFT/LoRA fine-tuning for Think-Anywhere")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to SFT dataset (JSONL)")
    parser.add_argument("--output_dir", type=str, default="results/sft", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    # QLoRA / 4-bit bitsandbytes flags (gated — off by default to preserve CPU smoke)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model with 4-bit quantization via bitsandbytes (requires GPU)")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Compute dtype for 4-bit quantization")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4",
                        choices=["fp4", "nf4"],
                        help="Quantization type for 4-bit (fp4 or nf4)")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true",
                        help="Use double quantization for 4-bit")
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA alpha (defaults to lora_rank * 2 if not set)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dataset without training")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: tiny-gpt2, 5 steps, CPU-safe, no real training")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max training steps (None = full epochs). Overrides epochs in smoke mode.")
    parser.add_argument("--limit_samples", type=int, default=None,
                        help="Limit dataset to N examples for fast iteration")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu"],
                        help="Device for training: auto (GPU if available) or cpu")
    return parser.parse_args()


def _format_instruction_response(instruction: str, response: str) -> str:
    """Format instruction/response pair into a TRL-compatible `text` field."""
    return (
        f"### Instruction:\n{instruction.strip()}\n\n"
        f"### Response:\n{response.strip()}\n"
    )


def _normalize_sft_record(record: dict) -> dict:
    """Normalize an SFT record to TRL-compatible format.

    - Records with `text` field: returned unchanged (legacy format).
    - Records with `instruction` + `response`: normalized to single `text` field.
    - Records with only `instruction` (no `response`): raise ValueError.
    - Other schemas: raise ValueError.
    """
    if "text" in record:
        # Legacy TRL format (possibly with extra fields) — pass through unchanged
        return record
    if "instruction" in record and "response" in record:
        return {"text": _format_instruction_response(record["instruction"], record["response"])}
    if "instruction" in record and "response" not in record:
        raise ValueError(
            f"Unsupported SFT record schema: record has `instruction` but no `response`. "
            f"Each SFT record must have either `text` (legacy TRL format) or both "
            f"`instruction` and `response` fields."
        )
    raise ValueError(
        f"Unsupported SFT record schema: record has keys {list(record.keys())}. "
        f"Expected either `text` (legacy TRL format) or `instruction` + `response`."
    )


def _load_dataset(path: str, limit: int | None = None):
    """Load JSONL dataset with normalization to TRL-compatible `text` format.

    Handles two schemas:
    - Legacy TRL format: records with only a `text` field — returned unchanged.
    - Phase 2 format: records with `instruction` + `response` — normalized to `text`.

    Raises ValueError for malformed records before model load.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    # Try UTF-8 first, fall back to cp1252 on Windows for internal TRL chat template files
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            if enc == "latin-1":
                raise
            continue
    records = [json.loads(line) for line in lines if line.strip()]
    records = [_normalize_sft_record(r) for r in records]
    if limit is not None:
        records = records[:limit]
    return records


def main():
    args = parse_args()

    print(f"Config: model={args.model_name}, dataset={args.dataset_path}, output={args.output_dir}")
    print(f"Training: epochs={args.epochs}, max_seq_length={args.max_seq_length}, lr={args.learning_rate}")
    print(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha or args.lora_rank * 2}, dropout={args.lora_dropout}, grad_accum={args.gradient_accumulation_steps}, per_device_bs={args.per_device_train_batch_size}")
    if args.load_in_4bit:
        print(f"4-bit: compute_dtype={args.bnb_4bit_compute_dtype}, quant_type={args.bnb_4bit_quant_type}, double_quant={args.bnb_4bit_use_double_quant}")
    else:
        print("4-bit: disabled (full precision)")

    dataset = _load_dataset(args.dataset_path, limit=args.limit_samples)
    print(f"Loaded {len(dataset)} examples from dataset.")

    # Smoke mode: override to tiny-gpt2 on CPU for fast local validation
    # NEVER enable bitsandbytes in smoke — it requires CUDA
    if args.smoke:
        print(f"[SMOKE] max_steps={args.max_steps or 5}, device={args.device}")
        args.model_name = "openai-community/gpt2"
        args.max_steps = args.max_steps or 5
        args.load_in_4bit = False  # force off — smoke is CPU-only
        # Force CPU on Windows; on Linux smoke can use auto
        if sys.platform == "win32" or args.device == "cpu":
            actual_device = "cpu"
        else:
            actual_device = "auto"
        print(f"[SMOKE] using tiny-gpt2, device={actual_device}")

    if args.dry_run or args.smoke:
        print("Dry run mode: skipping model load and training." if args.dry_run else "Smoke mode: skipping real training.")
        if args.model_name.lower() != "mock":
            print("Would load model and tokenizer here.")
        print("Dry run complete." if args.dry_run else "Smoke complete.")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # TRL >= 0.8: use processing_class instead of deprecated tokenizer arg
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Build quantization config if 4-bit loading is requested
    quantization_config = None
    if args.load_in_4bit:
        import torch
        compute_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(args.bnb_4bit_compute_dtype, torch.bfloat16)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        print(f"[4BIT] BitsAndBytesConfig: compute_dtype={args.bnb_4bit_compute_dtype}, "
              f"quant_type={args.bnb_4bit_quant_type}, double_quant={args.bnb_4bit_use_double_quant}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank * 2
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        save_strategy="epoch",
        logging_steps=10,
    )

    dataset = Dataset.from_list(dataset)

    # TRL >= 0.8: processing_class replaces deprecated tokenizer kwarg
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
