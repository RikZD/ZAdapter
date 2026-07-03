# ZAdapter — TPU-Native Bottleneck Adapters for Efficient LLM Fine-Tuning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Train large language models on free Kaggle TPU — no GPU, no CUDA, no `bitsandbytes` required.

---

## Why ZAdapter?

LoRA was designed around CUDA. Libraries like `bitsandbytes` (used for quantization
in most LoRA/QLoRA workflows) are CUDA-only and don't run on TPU. Getting standard
PEFT-style LoRA to behave correctly under `torch_xla`'s static-graph execution model
is also non-trivial — merge/unmerge logic and dynamic weight buffers don't map
cleanly onto XLA's compilation model.

ZAdapter takes a different approach: a simple bottleneck adapter (down-projection →
activation → up-projection, injected directly into the forward path of each
transformer block) that avoids merge logic entirely and plays well with XLA's
static graph. It was built and tested specifically for **free TPU access on
Kaggle**, so people without GPU access can still fine-tune LLMs.

---

## Tested Compatibility

All models below were tested end-to-end on **Kaggle TPU v5e-8**: adapter
injection → freeze verification → forward + backward pass.

| Model | Total Params | Trainable Params | Trainable % | Status |
|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-360M | 413M | 3.99M | 0.9675% | ✅ PASS |
| Qwen/Qwen2.5-0.5B | 633M | 2.80M | 0.4419% | ✅ PASS |
| Qwen/Qwen2.5-1.5B | 1.78B | 5.59M | 0.3137% | ✅ PASS |
| meta-llama/Llama-3.2-1B (tested via `unsloth/Llama-3.2-1B` mirror) | 1.50B | 4.26M | 0.2835% | ✅ PASS |

**Not yet tested:**
- Mixture-of-Experts architectures (DeepSeek-V2/V3, Mixtral, Qwen-MoE) — the
  auto-injection logic looks for `self_attn`/`mlp` submodules, which doesn't
  match MoE expert-routing structures. Likely won't inject correctly out of
  the box.
- Models above ~2B params — should work in principle (single-chip TPU v5e-8
  has 16GB HBM), but untested. Larger models will likely need multi-chip
  sharding, which `ZAdapterTrainer` doesn't implement yet.
- Non-Llama-style architectures (e.g. GPT-2/GPT-Neo's `attn` naming, Falcon's
  merged attention+MLP blocks).

If you test ZAdapter on an architecture not listed here, contributions to the
compatibility table are welcome — see `tests/test_compat.py`.

---

## Installation

From source (not yet on PyPI):

```bash
git clone https://github.com/rikzd/ZAdapter.git
cd ZAdapter
pip install -e .
```

On Kaggle, upload the repo as a Dataset and add it to your notebook — see
`ZAdapter-Kaggle-SmolLM2.ipynb` for a full working example.

> ⚠️ **Do not `pip install torch_xla` manually on Kaggle TPU notebooks.**
> Kaggle's TPU runtime already ships a compatible, pre-linked version. Installing
> a different version from PyPI can cause `undefined symbol` errors from binary
> incompatibility between `torch` and `torch_xla`. If you hit this, restart the
> session instead of reinstalling.

---

## Quickstart

### 1. Inject adapters

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from z_adapter import inject_adapter, print_trainable_parameters

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# On TPU: pass device= directly, this handles the XLA freeze gotcha automatically
import torch_xla.core.xla_model as xm
device = xm.xla_device()

model = inject_adapter(model, r=64, device=device)
print_trainable_parameters(model)
# Total: 632,964,480 | Trainable: 2,797,056 (0.4419%)
```

### 2. Train

```python
from datasets import load_dataset
from z_adapter import ZAdapterTrainer

dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

trainer = ZAdapterTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    output_dir="./z-adapter-output",
    batch_size=8,
    learning_rate=1e-4,
    num_epochs=3,
)
trainer.train()
```

`ZAdapterTrainer` automatically re-applies the freeze mask and runs a sanity
check right after moving the model to device — see **Known TPU Gotcha** below
for why this matters.

### 3. Inference

```python
from z_adapter.utils import load_adapter

load_adapter(model, "./z-adapter-output/adapter_final.pt")
model.eval()

inputs = tokenizer("Explain quantum computing:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

---

## How It Works

```
Transformer Block:
  input → [self_attn wrapped in ZAdapter] → [mlp wrapped in ZAdapter] → output
```

Unlike LoRA (which decomposes weight updates as W' = W + (α/r)·BA and requires
merge/unmerge), ZAdapter wraps the *existing* `self_attn` and `mlp` submodules
so their output is passed through a small bottleneck FFN
(`down_proj → activation → up_proj`) before continuing. The base model stays
completely untouched and frozen — only the adapter's own parameters are
trainable. This means:

- No weight merging, no buffer manipulation
- The computational graph is static — XLA sees the same shape/path every step
- Freezing is a straightforward parameter filter, not a merge-aware operation

### vs LoRA

| Aspect | LoRA | ZAdapter |
|---|---|---|
| Update mechanism | Low-rank matrix (W + BA) | Bottleneck FFN wrapper |
| Merge required | Yes | No |
| XLA graph | Can require re-tracing on merge | Static |
| Quantization (QLoRA-style) | Needs `bitsandbytes` (CUDA-only) | N/A — not implemented yet |

---

## Known TPU Gotcha: `requires_grad` Reset on `.to(device)`

Some `torch_xla` versions reset `requires_grad` flags when transferring a
model to the XLA device, even after freezing the base model correctly. If you
build your own training loop instead of using `ZAdapterTrainer`, always
re-apply the freeze mask **after** `.to(device)`:

```python
model = model.to(device)
refreeze_after_device_transfer(model)   # re-applies the freeze mask
verify_frozen(model)                     # raises immediately if it's still wrong
```

Skipping this can silently train the entire base model instead of just the
adapters — no error, just much slower training and OOM risk, with no obvious
cause. `inject_adapter(model, device=...)` and `ZAdapterTrainer` both handle
this automatically.

---

## Roadmap

- [x] Core bottleneck adapter + TPU-safe injection
- [x] Compatibility validated on 4 architectures (TPU v5e-8)
- [ ] QLoRA-style quantization for TPU (separate project, in progress)
- [ ] MoE architecture support (DeepSeek, Mixtral-style expert routing)
- [ ] Multi-chip sharding for models >2B
- [ ] JAX/Pallas backend — planned pending community demand for further
      kernel-level optimization; not started

---

## Contributing

Compatibility reports on untested architectures, bug reports, and PRs are
welcome. Run `tests/test_compat.py` before submitting changes that touch
`core.py` or `trainer.py`.

---

## License

MIT License
