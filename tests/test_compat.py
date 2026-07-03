"""
ZAdapter Compatibility Test
============================
Test inject_adapter() + freeze logic across small Llama-family models.
Run this on Kaggle TPU v5e-8 before publishing v0.1.1.

Usage:
    python test_compat.py --model Qwen/Qwen2.5-0.5B --device tpu
    python test_compat.py --model meta-llama/Llama-3.2-1B --device tpu
    python test_compat.py --model HuggingFaceTB/SmolLM2-360M --device tpu
"""

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "..")
from z_adapter import (
    inject_adapter,
    get_trainable_params,
    verify_frozen,
    print_trainable_parameters,
    count_adapters,
)


MODELS_TO_TEST = [
    "HuggingFaceTB/SmolLM2-360M",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "meta-llama/Llama-3.2-1B",
]


def get_device(device_arg: str):
    if device_arg == "tpu":
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    elif device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def run_single_test(model_name: str, device_arg: str, r: int = 32) -> dict:
    """Load model, inject adapter, verify freeze, run one forward+backward pass."""
    result = {"model": model_name, "status": "FAIL", "error": None, "trainable_pct": None, "num_adapters": None}

    try:
        print(f"\n{'='*60}\nTesting: {model_name}\n{'='*60}")

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        print(f"  Loaded in {time.time() - t0:.1f}s")

        device = get_device(device_arg)

        # This is the exact call path that had the freeze bug — device=
        # handles inject + move + refreeze in one call.
        model = inject_adapter(model, r=r, device=device)

        num_adapters = count_adapters(model)
        result["num_adapters"] = num_adapters
        if num_adapters == 0:
            raise RuntimeError("inject_adapter() found 0 injection points — architecture not recognized")

        # This is the critical check: if this raises, the freeze bug is back
        # for this architecture.
        pct = verify_frozen(model, max_trainable_pct=5.0)
        result["trainable_pct"] = pct

        # Sanity: actually run a forward + backward pass, not just check flags
        dummy_input = tokenizer("Hello, this is a test.", return_tensors="pt").to(device)
        params_before = get_trainable_params(model)
        optimizer = torch.optim.AdamW(params_before, lr=1e-4)

        outputs = model(**dummy_input, labels=dummy_input["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if device_arg == "tpu":
            import torch_xla.core.xla_model as xm
            xm.mark_step()

        print(f"  Forward+backward OK. Loss: {loss.item():.4f}")
        print_trainable_parameters(model)

        result["status"] = "PASS"

    except Exception as e:
        result["error"] = str(e)
        print(f"  FAILED: {e}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Single model to test (skip full suite)")
    parser.add_argument("--device", type=str, default="tpu", choices=["tpu", "cuda", "cpu"])
    parser.add_argument("--r", type=int, default=32, help="Adapter rank")
    args = parser.parse_args()

    models = [args.model] if args.model else MODELS_TO_TEST

    results = []
    for model_name in models:
        results.append(run_single_test(model_name, args.device, args.r))

    print(f"\n\n{'='*60}\nSUMMARY\n{'='*60}")
    for r in results:
        status_mark = "PASS" if r["status"] == "PASS" else "FAIL"
        pct_str = f"{r['trainable_pct']:.4f}%" if r["trainable_pct"] is not None else "N/A"
        adapters_str = str(r["num_adapters"]) if r["num_adapters"] is not None else "N/A"
        print(f"[{status_mark}] {r['model']:<40} trainable={pct_str:<12} adapters={adapters_str}")
        if r["error"]:
            print(f"       -> {r['error']}")

    failed = [r for r in results if r["status"] == "FAIL"]
    if failed:
        print(f"\n{len(failed)}/{len(results)} models FAILED. Fix before publishing.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} models PASSED.")


if __name__ == "__main__":
    main()
