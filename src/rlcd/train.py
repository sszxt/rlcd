"""LoRA + DPO fine-tune on the preference pairs from generate_preferences.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

from rlcd.model_utils import DEFAULT_MODEL, load_model_and_tokenizer

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--preferences", default=str(OUTPUTS_DIR / "preferences.jsonl"))
    parser.add_argument("--out-dir", default=str(OUTPUTS_DIR / "lora_adapter"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)
    tokenizer.padding_side = "right"  # DPOTrainer pads on the right during training

    dataset = load_dataset("json", data_files=args.preferences, split="train")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    dpo_config = DPOConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        beta=args.beta,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=512,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    print(f"saved LoRA adapter to {args.out_dir}")


if __name__ == "__main__":
    main()
