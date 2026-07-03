# ZAdapter — TPU-Native Bottleneck Adapter for Efficient LLM Fine-Tuning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Simple bottleneck FFN adapter. No merge. No freeze headache. Pure static graph.**

---

## 🚀 Features

- **Bottleneck Adapter** — FFN down-projection + up-projection (Houlsby et al. style)
- **TPU-Native** — Static graph, no merge/unmerge complexity, XLA-friendly
- **90%+ LoRA Equivalent** — Low-rank, few params (~0.5-3%), swappable adapters
- **Universal** — Works with any HuggingFace transformer (Llama, Qwen, Mistral, etc.)
- **Kaggle-Ready** — Compatible with TPU v5e-8

---

## 📦 Installation

```bash
pip install z-adapter
```

From source:
```bash
git clone https://github.com/rikzd/z-adapter.git
cd z-adapter
pip install -e .
```

---

## 🎯 Quick Start

### 1. Inject ZAdapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from z_adapter import inject_adapter, ZAdapterTrainer, print_trainable_parameters

# Load model
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")

# Inject ZAdapter (1 line!)
model = inject_adapter(model, r=64)

# Check parameters
print_trainable_parameters(model)
# Output: Total: 360M | Trainable: 7.8M (2.2%)
```

### 2. Train

```python
from datasets import load_dataset

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

### 3. Inference

```python
from z_adapter.utils import load_adapter

# Load adapter
load_adapter(model, "./z-adapter-output/adapter.pt")
model.eval()

# Generate
inputs = tokenizer("Explain quantum computing:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

---

## 🔧 How It Works

### Architecture

```
Transformer Block:
  input → Self-Attention → [ZAdapter] → MLP → [ZAdapter] → output
                         ↑              ↑
                         └─ bottleneck ─┘
                         [d_model → r → d_model]
```

### vs LoRA

| Aspect | LoRA | ZAdapter |
|--------|------|----------|
| Update type | Low-rank matrix | Bottleneck FFN |
| Params | ~0.5-1% | ~0.5-3% |
| Merge | Required (complex) | Not needed |
| Freeze logic | Complex | Simple (only adapters trainable) |
| XLA graph | Dynamic (merge) | Static (no merge) |
| Gradient flow | Tricky (buffer issue) | Straightforward |

---

## 📊 Benchmarks

Coming soon!

---

## 🛣️ Roadmap

- [x] Core adapter implementation
- [x] TPU trainer
- [x] Kaggle notebook
- [ ] Multi-adapter switching
- [ ] Prefix tuning hybrid
- [ ] Benchmark suite

---

## 📄 License

MIT License

---

> Built by ZNX Team
