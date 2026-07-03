import torch
import torch.nn as nn
from z_adapter.core import ZAdapter, inject_adapter, get_trainable_params

def test_zadapter():
    adapter = ZAdapter(d_model=960, r=64)
    x = torch.randn(2, 10, 960)
    out = adapter(x)
    assert out.shape == (2, 10, 960)
    print("✅ ZAdapter forward")

def test_inject():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 960})()
            self.layers = nn.ModuleList([nn.Module() for _ in range(4)])
            for i, layer in enumerate(self.layers):
                layer.self_attn = nn.Linear(960, 960)
                layer.mlp = nn.Linear(960, 960)

    model = DummyModel()
    model = inject_adapter(model, r=32)

    params = get_trainable_params(model)
    assert len(params) > 0
    print(f"✅ Injected adapters: {len(params)} params")

if __name__ == "__main__":
    test_zadapter()
    test_inject()
    print("✅ All tests passed")
