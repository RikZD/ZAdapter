"""
ZAdapter Trainer - TPU-Optimized Training Loop
===============================================

Simple training loop. No merge. No complexity. Just train adapters.
"""

import os
import torch
from torch.utils.data import DataLoader
from typing import Optional


def _is_tpu_available():
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        return True
    except ImportError:
        return False


class ZAdapterTrainer:
    """
    Trainer for ZAdapter on TPU or GPU.

    Args:
        model: Model with ZAdapter injected
        tokenizer: HF tokenizer
        dataset: HF dataset
        output_dir: Directory to save checkpoints
        batch_size: Batch size per device
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        max_seq_length: Max sequence length
        gradient_accumulation: Gradient accumulation steps
        warmup_steps: Warmup steps for LR scheduler
        max_grad_norm: Gradient clipping norm
        logging_steps: Print loss every N steps
        save_steps: Save checkpoint every N steps
    """

    def __init__(
        self,
        model,
        tokenizer,
        dataset,
        output_dir: str = "./z-adapter-output",
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        max_seq_length: int = 512,
        gradient_accumulation: int = 4,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        save_steps: int = 500,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = num_epochs
        self.max_seq_length = max_seq_length
        self.grad_accum = gradient_accumulation
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.save_steps = save_steps

        # Detect device
        self.use_tpu = _is_tpu_available()
        if self.use_tpu:
            import torch_xla.core.xla_model as xm
            self.device = xm.xla_device()
            print(f"[ZAdapter] TPU detected: {xm.xla_real_devices([self.device])}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[ZAdapter] Using device: {self.device}")

        self.model = self.model.to(self.device)

        # Re-apply freeze + sanity check — .to(device) on XLA can silently
        # reset requires_grad, which would make the whole base model trainable.
        from .utils import refreeze_after_device_transfer, verify_frozen
        refreeze_after_device_transfer(self.model)
        verify_frozen(self.model)

        # Optimizer: only adapter params
        from .core import get_trainable_params
        trainable = get_trainable_params(self.model)
        self.optimizer = torch.optim.AdamW(trainable, lr=self.lr)

        self.scheduler = None
        self.global_step = 0
        self.epoch = 0

    def _prepare_dataset(self):
        """Tokenize and format dataset."""
        def tokenize_fn(examples):
            if "text" in examples:
                texts = examples["text"]
            elif "conversations" in examples:
                # ShareGPT-style format: list of {"role"/"from": ..., "value"/"content": ...}
                texts = []
                for conv in examples["conversations"]:
                    turns = []
                    for turn in conv:
                        role = turn.get("role") or turn.get("from", "unknown")
                        value = turn.get("value") or turn.get("content", "")
                        turns.append(f"{role}: {value}")
                    texts.append("\n".join(turns))
            elif "instruction" in examples and "output" in examples:
                texts = [
                    f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                    for inst, out in zip(examples["instruction"], examples["output"])
                ]
            elif "input" in examples and "output" in examples:
                texts = [
                    f"### Input:\n{inp}\n\n### Response:\n{out}"
                    for inp, out in zip(examples["input"], examples["output"])
                ]
            else:
                raise ValueError(
                    "Dataset must have 'text', 'conversations' (ShareGPT), "
                    "'instruction'/'output', or 'input'/'output' columns"
                )

            return self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
            )

        tokenized = self.dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=self.dataset.column_names,
        )
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return tokenized

    def _setup_scheduler(self, num_training_steps):
        """Setup learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - self.warmup_steps))
            )

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """Main training loop."""
        self.model.train()

        print("[ZAdapter] Preparing dataset...")
        tokenized = self._prepare_dataset()

        total_steps = len(tokenized) // self.batch_size // self.grad_accum * self.epochs
        self._setup_scheduler(total_steps)

        print(f"[ZAdapter] Training for {self.epochs} epochs (~{total_steps} steps)")
        print(f"[ZAdapter] Gradient accumulation: {self.grad_accum}")

        for epoch in range(self.epochs):
            self.epoch = epoch
            print(f"\n=== Epoch {epoch + 1}/{self.epochs} ===")

            dataloader = DataLoader(tokenized, batch_size=self.batch_size, shuffle=True, drop_last=True)

            if self.use_tpu:
                import torch_xla.distributed.parallel_loader as pl
                data_iter = pl.ParallelLoader(dataloader, [self.device]).per_device_loader(self.device)
            else:
                data_iter = dataloader

            epoch_loss = 0.0
            num_batches = 0

            for batch in data_iter:
                loss = self._training_step(batch)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

            self.save_checkpoint(suffix=f"epoch_{epoch + 1}")

        print("\n[ZAdapter] Training complete!")
        self.save_checkpoint(suffix="final")

    def _training_step(self, batch):
        """Single training step with gradient accumulation."""

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss / self.grad_accum
        loss.backward()

        if (self.global_step + 1) % self.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm
            )

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler:
                self.scheduler.step()

            if self.global_step % self.logging_steps == 0:
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.lr
                print(f"Step {self.global_step} | Loss: {loss.item() * self.grad_accum:.4f} | LR: {current_lr:.2e}")

            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(suffix=f"step_{self.global_step}")

            if self.use_tpu:
                import torch_xla.core.xla_model as xm
                xm.mark_step()

        self.global_step += 1
        return loss.item() * self.grad_accum

    def save_checkpoint(self, suffix: str = ""):
        """Save adapter checkpoint (adapter weights only, not the full model)."""
        from .utils import save_adapter

        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"adapter_{suffix}.pt" if suffix else "adapter.pt"
        path = os.path.join(self.output_dir, filename)

        save_adapter(self.model, path)
        print(f"  [Saved: {path}]")
