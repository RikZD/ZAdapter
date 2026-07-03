"""
ZAdapter Utils
==============
Helper functions: save/load adapter weights, dan safety checks buat TPU/XLA.
"""

import torch


def save_adapter(model, path: str, adapter_keyword: str = "z_adapter"):
    """Save cuma adapter weights (bukan full model)."""
    adapter_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if adapter_keyword in name
    }
    torch.save(adapter_state, path)


def load_adapter(model, path: str, device=None):
    """Load adapter weights ke model yang udah di-inject sebelumnya."""
    adapter_state = torch.load(path, map_location="cpu")
    model_state_keys = model.state_dict().keys()

    filtered = {k: v for k, v in adapter_state.items() if k in model_state_keys}
    missing = [k for k in adapter_state.keys() if k not in model_state_keys]

    if missing:
        print(f"[ZAdapter] Warning: {len(missing)} adapter keys not found in model: {missing[:5]}...")

    model.load_state_dict(filtered, strict=False)

    if device is not None:
        model = model.to(device)

    return model


def refreeze_after_device_transfer(model, adapter_keyword: str = "z_adapter"):
    """
    Re-apply freeze mask setelah model.to(device).

    WAJIB dipanggil setelah .to(xla_device()) di TPU — beberapa versi
    torch_xla bisa reset requires_grad ke True untuk semua parameter
    saat transfer tensor ke XLA device, yang bikin base model ikut ke-train.
    """
    for name, param in model.named_parameters():
        param.requires_grad = adapter_keyword in name


def verify_frozen(model, max_trainable_pct: float = 5.0, verbose: bool = True) -> float:
    """
    Sanity check: pastiin cuma adapter yang trainable.

    Panggil ini setelah model.to(device) + refreeze_after_device_transfer().
    Raise RuntimeError kalau trainable % di luar batas wajar — ini nangkep
    bug freeze yang silent-fail (training jalan normal tapi semua param ke-train,
    OOM atau training jadi lambat tanpa error jelas).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100 * trainable / total if total > 0 else 0.0

    if trainable_pct > max_trainable_pct:
        raise RuntimeError(
            f"[ZAdapter] Freeze check FAILED: {trainable_pct:.2f}% trainable "
            f"(expected <{max_trainable_pct}%). Ini biasanya kejadian karena "
            f"requires_grad ke-reset setelah model.to(device) di XLA. "
            f"Fix: panggil refreeze_after_device_transfer(model) setelah .to(device)."
        )

    if verbose:
        print(f"[ZAdapter] Freeze check passed: {trainable_pct:.4f}% trainable "
              f"({trainable:,} / {total:,} params)")

    return trainable_pct


def print_trainable_parameters(model):
    """Print ringkasan total vs trainable params."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total if total > 0 else 0.0
    print(f"Total: {total:,} | Trainable: {trainable:,} ({pct:.4f}%)")
