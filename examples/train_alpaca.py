#!/usr/bin/env python3
"""ZAdapter Example: Fine-tune on Alpaca Dataset"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from z_adapter import inject_adapter, ZAdapterTrainer, print_trainable_parameters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--r", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="./z-adapter-output")
    args = parser.parse_args()

    print("=" * 60)
    print("ZAdapter Training")
    print("=" * 60)

    # Load
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Inject
    model = inject_adapter(model, r=args.r)
    print_trainable_parameters(model)

    # Dataset
    dataset = load_dataset(args.dataset, split="train[:1000]")

    # Train
    trainer = ZAdapterTrainer(
        model=model, tokenizer=tokenizer, dataset=dataset,
        output_dir=args.output_dir, batch_size=args.batch_size,
        learning_rate=args.lr, num_epochs=args.epochs,
    )
    trainer.train()

    print("\n✅ Done!")

if __name__ == "__main__":
    main()
