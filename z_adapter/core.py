"""
ZAdapter Core - TPU-Native Bottleneck Adapter
==============================================

Simple bottleneck FFN adapter. No merge. No freeze headache. Pure static graph.

Architecture:
    input -> [d_model] -> down_proj [r] -> activation -> up_proj [d_model] -> output

    residual: output = input + adapter(input)

~90%+ functionally equivalent to LoRA:
    - Low-rank bottleneck (r << d_model)
    - Few trainable params (~0.5-3% of model)
    - Can swap adapters
    - Universal model injection

TPU advantages over LoRA:
    - No merge/unmerge complexity
    - Static graph (XLA sees 1 path)
    - Simple gradient flow
    - No weight buffer manipulation
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ZAdapter(nn.Module):
    """
    Bottleneck adapter layer for TPU.

    Args:
        d_model: Hidden dimension of transformer
        r: Bottleneck rank (default: 64)
        dropout: Dropout probability (default: 0.0)
        activation: Activation function (default: GELU)
    """

    def __init__(
        self,
        d_model: int,
        r: int = 64,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()

        self.d_model = d_model
        self.r = r

        self.down_proj = nn.Linear(d_model, r)
        self.up_proj = nn.Linear(r, d_model)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Init: down small, up zero -> adapter starts as identity function
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, d_model]
        Returns:
            [batch, seq, d_model] - input + adapter(input)
        """
        residual = x
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up_proj(h)
        return residual + h

    def extra_repr(self):
        return f"d_model={self.d_model}, r={self.r}"


class _AdapterWrapper(nn.Module):
    """
    Wraps an existing submodule (e.g. self_attn or mlp) so its output is
    passed through a ZAdapter before being returned.

    This is what actually puts the adapter on the computational graph.
    Just doing model.add_module("z_adapter_x", ZAdapter(...)) without this
    wrapper leaves the adapter as an unused sibling -- it never gets called
    during forward(), so backward() has nothing to build gradients from.
    """

    def __init__(self, wrapped_module: nn.Module, d_model: int, r: int, dropout: float = 0.0):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.z_adapter = ZAdapter(d_model=d_model, r=r, dropout=dropout)

        # Match dtype of the wrapped module so hidden_states (e.g. bfloat16)
        # don't mismatch against the adapter's default float32 weights.
        wrapped_dtype = next(wrapped_module.parameters(), None)
        if wrapped_dtype is not None:
            self.z_adapter = self.z_adapter.to(dtype=wrapped_dtype.dtype)

    def forward(self, *args, **kwargs):
        output = self.wrapped_module(*args, **kwargs)

        # Many HF attention modules return a tuple (hidden_states, attn_weights, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states = self.z_adapter(hidden_states)
            return (hidden_states,) + output[1:]

        return self.z_adapter(output)

def _freeze_base_model(model, adapter_keyword: str = "z_adapter"):
    """Internal: set requires_grad based on adapter membership."""
    for name, param in model.named_parameters():
        param.requires_grad = adapter_keyword in name


def inject_adapter(
    model,
    r: int = 64,
    target_locations: Optional[List[str]] = None,
    dropout: float = 0.0,
    device=None,
):
    """
    Inject ZAdapter into transformer model and freeze the base model.

    For Llama-style models: inject after attention and after MLP in each layer.

    Args:
        model: HF transformer model
        r: Bottleneck rank
        target_locations: Where to insert (reserved, currently auto-detect only)
        dropout: Dropout probability
        device: Optional device (e.g. xm.xla_device()). If provided, the model
            is moved to this device AFTER injection, and the freeze mask is
            automatically re-applied — this is the safe way to combine
            inject_adapter() with TPU/XLA, since .to(device) on XLA can reset
            requires_grad flags.

    Returns:
        Model with adapters injected (base params frozen)
    """
    if target_locations is None:
        target_locations = ["attention", "mlp", "feed_forward"]

    injected = 0
    d_model = None

    if hasattr(model, "config"):
        d_model = getattr(model.config, "hidden_size", None)

    for name, module in model.named_modules():
        # Pattern: Llama/Qwen/Mistral-style decoder layer, e.g. model.layers.0
        if "layers" in name and name.count(".") == 2:
            layer_idx = name.split(".")[-1]
            if layer_idx.isdigit():
                if d_model is None:
                    for child in module.modules():
                        if isinstance(child, nn.Linear):
                            d_model = child.out_features
                            break

                if d_model:
                    attn_attr = "self_attn" if hasattr(module, "self_attn") else (
                        "attention" if hasattr(module, "attention") else None
                    )
                    if attn_attr and not isinstance(getattr(module, attn_attr), _AdapterWrapper):
                        original = getattr(module, attn_attr)
                        setattr(module, attn_attr, _AdapterWrapper(original, d_model, r, dropout))
                        injected += 1

                    mlp_attr = "mlp" if hasattr(module, "mlp") else (
                        "feed_forward" if hasattr(module, "feed_forward") else None
                    )
                    if mlp_attr and not isinstance(getattr(module, mlp_attr), _AdapterWrapper):
                        original = getattr(module, mlp_attr)
                        setattr(module, mlp_attr, _AdapterWrapper(original, d_model, r, dropout))
                        injected += 1

    if injected == 0:
        injected = _manual_inject(model, r, dropout)

    # Freeze base model, only adapters trainable
    _freeze_base_model(model)

    if device is not None:
        model = model.to(device)
        # XLA .to(device) can reset requires_grad -> re-apply freeze mask
        _freeze_base_model(model)

    print(f"[ZAdapter] Injected {injected} adapters")
    return model


def _manual_inject(model, r, dropout):
    """Manual injection fallback for models where auto-detect fails."""
    injected = 0
    d_model = None

    for module in model.modules():
        if isinstance(module, nn.Linear):
            d_model = module.out_features
            break

    if d_model is None:
        print("[ZAdapter] ERROR: Could not detect d_model")
        return 0

    for name, module in model.named_modules():
        if name.endswith(".layers") or name.endswith(".h") or name.endswith(".blocks"):
            for child_name, child in module.named_children():
                if child_name.isdigit():
                    layer = child
                    attn_attr = "self_attn" if hasattr(layer, "self_attn") else (
                        "attention" if hasattr(layer, "attention") else None
                    )
                    if attn_attr and not isinstance(getattr(layer, attn_attr), _AdapterWrapper):
                        original = getattr(layer, attn_attr)
                        setattr(layer, attn_attr, _AdapterWrapper(original, d_model, r, dropout))
                        injected += 1

                    mlp_attr = "mlp" if hasattr(layer, "mlp") else (
                        "feed_forward" if hasattr(layer, "feed_forward") else None
                    )
                    if mlp_attr and not isinstance(getattr(layer, mlp_attr), _AdapterWrapper):
                        original = getattr(layer, mlp_attr)
                        setattr(layer, mlp_attr, _AdapterWrapper(original, d_model, r, dropout))
                        injected += 1

    return injected


def get_trainable_params(model) -> List[torch.nn.Parameter]:
    """
    Return only adapter parameters, re-applying the freeze mask first.

    Re-applying here (not just filtering) matters on XLA: if requires_grad
    was reset by a prior .to(device) call, this restores the correct state
    before handing params to the optimizer.
    """
    _freeze_base_model(model)
    return [p for name, p in model.named_parameters() if "z_adapter" in name]


def count_adapters(model) -> int:
    """Count number of ZAdapter modules in model (including those nested inside _AdapterWrapper)."""
    count = 0
    for module in model.modules():
        if isinstance(module, ZAdapter):
            count += 1
    return count


def count_wrapped_modules(model) -> int:
    """Count how many original modules (self_attn/mlp) got wrapped with an adapter."""
    count = 0
    for module in model.modules():
        if isinstance(module, _AdapterWrapper):
            count += 1
    return count
