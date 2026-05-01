from __future__ import annotations

import numpy as np

from .runtime import as_numpy, gpu_stack_ready, require_gpu_stack


def layernorm_reference(x, eps: float = 1e-5):
    x = as_numpy(x).astype(np.float64, copy=False)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def layernorm_triton(x, eps: float = 1e-5):
    torch, triton, tl = require_gpu_stack()
    if not torch.is_tensor(x):
        raise TypeError("layernorm_triton expects a torch tensor.")
    if not x.is_cuda:
        raise ValueError("layernorm_triton expects a CUDA tensor.")
    if x.ndim != 2:
        raise ValueError("layernorm_triton is a 2D starter kernel.")

    x = x.contiguous().to(torch.float32)
    rows, cols = x.shape
    if cols > 1024:
        raise ValueError("starter layernorm expects at most 1024 columns per row")
    out = torch.empty_like(x)

    @triton.jit
    def kernel(x_ptr, y_ptr, stride_xm, stride_xn, stride_ym, stride_yn, n_cols, eps, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        values = tl.load(x_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0)
        mean = tl.sum(values, axis=0) / n_cols
        centered = values - mean
        var = tl.sum(centered * centered, axis=0) / n_cols
        inv_std = tl.math.rsqrt(var + eps)
        tl.store(y_ptr + row * stride_ym + cols * stride_yn, centered * inv_std, mask=mask)

    block_size = 1 << (cols - 1).bit_length()
    grid = lambda meta: (rows,)
    kernel[grid](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


def layernorm(x, eps: float = 1e-5, backend: str = "auto"):
    if backend == "numpy":
        return layernorm_reference(x, eps=eps)
    if backend == "triton":
        return layernorm_triton(x, eps=eps)
    if backend == "auto":
        try:
            import torch
        except Exception:
            return layernorm_reference(x, eps=eps)
        if torch.is_tensor(x) and x.is_cuda and gpu_stack_ready():
            return layernorm_triton(x, eps=eps)
    return layernorm_reference(x, eps=eps)


if __name__ == "__main__":
    sample = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    print(layernorm(sample))
