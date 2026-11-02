"""Sanity-check the pointact environment: imports, GPU, and the CUDA extensions.

Run with: pixi run check-env
"""

import importlib
import sys

ok = True


def check(name, fn):
    global ok
    try:
        msg = fn()
        print(f"[ok]   {name}" + (f": {msg}" if msg else ""))
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def _torch():
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    x = torch.randn(256, 256, device="cuda")
    (x @ x).sum().item()
    return f"{torch.__version__}, {torch.cuda.get_device_name(0)}, arch list {torch.cuda.get_arch_list()}"


def _nvrtc():
    # Integer prod goes through torch's NVRTC JIT; on GB10 (sm_121) this needs NVRTC >= 12.9.
    import torch

    out = torch.tensor([[1, 2, 3]], device="cuda").prod(-1)
    assert out.tolist() == [6]
    return ""


def _version(module):
    def fn():
        return importlib.import_module(module).__version__

    return fn


def _spconv():
    import spconv.pytorch as spconv
    import torch

    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2]], dtype=torch.int32, device="cuda")
    feats = torch.randn(3, 16, device="cuda")
    x = spconv.SparseConvTensor(feats, coords, spatial_shape=[4, 4, 4], batch_size=1)
    conv = spconv.SubMConv3d(16, 32, kernel_size=3, indice_key="t").cuda()
    out = conv(x)
    assert out.features.shape == (3, 32)
    return importlib.import_module("spconv").__version__


def _torch_scatter():
    import torch
    import torch_scatter

    out = torch_scatter.scatter_max(torch.arange(4.0, device="cuda"), torch.tensor([0, 0, 1, 1], device="cuda"))[0]
    assert out.tolist() == [1.0, 3.0]
    return torch_scatter.__version__


def _flash_attn():
    import flash_attn
    import torch
    from flash_attn import flash_attn_func

    q = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.bfloat16)
    flash_attn_func(q, q, q)
    return flash_attn.__version__


check("torch + GPU", _torch)
check("torch NVRTC JIT", _nvrtc)
for mod in ("transformers", "lerobot", "numpy", "open3d"):
    check(mod, _version(mod))
check("spconv", _spconv)
check("torch_scatter", _torch_scatter)
check("flash_attn", _flash_attn)
check("pointact", lambda: importlib.import_module("pointact") and "")

sys.exit(0 if ok else 1)
