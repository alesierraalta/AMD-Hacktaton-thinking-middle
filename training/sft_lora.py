import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="SFT/LoRA fine-tuning for Think-Anywhere")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to SFT dataset (JSONL)")
    parser.add_argument("--output_dir", type=str, default="results/sft", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dataset without training")
    return parser.parse_args()


def _load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    args = parse_args()

    print(f"Config: model={args.model_name}, dataset={args.dataset_path}, output={args.output_dir}")
    print(f"Training: epochs={args.epochs}, max_seq_length={args.max_seq_length}, lr={args.learning_rate}")
    print(f"LoRA: rank={args.lora_rank}, grad_accum={args.gradient_accumulation_steps}")

    dataset = _load_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} examples from dataset.")

    if args.dry_run:
        print("Dry run mode: skipping model load and training.")
        if args.model_name.lower() != "mock":
            print("Would load model and tokenizer here.")
        print("Dry run complete.")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        save_strategy="epoch",
        logging_steps=10,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
