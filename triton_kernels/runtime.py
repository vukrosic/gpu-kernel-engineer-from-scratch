from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import numpy as np


def has_torch() -> bool:
    return find_spec("torch") is not None


def has_triton() -> bool:
    return find_spec("triton") is not None


def has_cuda_runtime() -> bool:
    if not has_torch():
        return False
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def gpu_stack_ready() -> bool:
    return has_torch() and has_triton() and has_cuda_runtime()


def is_torch_cuda_tensor(value: Any) -> bool:
    if not has_torch():
        return False
    try:
        import torch
    except Exception:
        return False
    return torch.is_tensor(value) and value.is_cuda


def is_torch_cuda_tensor_pair(a: Any, b: Any) -> bool:
    return is_torch_cuda_tensor(a) and is_torch_cuda_tensor(b)


def as_numpy(value: Any) -> np.ndarray:
    if has_torch():
        try:
            import torch
        except Exception:
            pass
        else:
            if torch.is_tensor(value):
                return value.detach().cpu().numpy()
    return np.asarray(value)


def require_gpu_stack():
    if not has_torch():
        raise RuntimeError("PyTorch is required for the GPU path.")
    if not has_triton():
        raise RuntimeError("Triton is required for the GPU path.")
    if not has_cuda_runtime():
        raise RuntimeError("A CUDA-capable PyTorch runtime is required.")
    import torch
    import triton
    import triton.language as tl

    return torch, triton, tl
