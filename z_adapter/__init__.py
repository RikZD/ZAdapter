"""
ZAdapter - TPU-Native Adapter for Efficient LLM Fine-Tuning
==========================================================

Lightweight bottleneck adapter optimized for Google TPU (XLA/MXU).
~90%+ functionally equivalent to LoRA with a simpler, XLA-friendly architecture.

Quick Start (GPU or CPU):
    from z_adapter import inject_adapter, ZAdapterTrainer

    model = inject_adapter(model, r=64)
    trainer = ZAdapterTrainer(model, tokenizer, dataset)
    trainer.train()

Quick Start (TPU):
    import torch_xla.core.xla_model as xm
    from z_adapter import inject_adapter, ZAdapterTrainer

    device = xm.xla_device()
    model = inject_adapter(model, r=64, device=device)  # handles XLA freeze gotcha automatically
    trainer = ZAdapterTrainer(model, tokenizer, dataset)
    trainer.train()
"""

from .core import (
    ZAdapter,
    inject_adapter,
    get_trainable_params,
    count_adapters,
    count_wrapped_modules,
)

from .trainer import ZAdapterTrainer

from .utils import (
    save_adapter,
    load_adapter,
    print_trainable_parameters,
    refreeze_after_device_transfer,
    verify_frozen,
)

__version__ = "0.1.0"
__all__ = [
    "ZAdapter",
    "inject_adapter",
    "get_trainable_params",
    "count_adapters",
    "count_wrapped_modules",
    "ZAdapterTrainer",
    "save_adapter",
    "load_adapter",
    "print_trainable_parameters",
    "refreeze_after_device_transfer",
    "verify_frozen",
]
