from __future__ import annotations

import numpy as np

from .runtime import as_numpy, gpu_stack_ready, require_gpu_stack


def reduce_sum_reference(x, axis: int = -1):
    return as_numpy(x).sum(axis=axis)


def reduce_sum_triton(x):
    torch, triton, tl = require_gpu_stack()
    if not torch.is_tensor(x):
        raise TypeError("reduce_sum_triton expects a torch tensor.")
    if not x.is_cuda:
        raise ValueError("reduce_sum_triton expects a CUDA tensor.")
    if x.ndim != 2:
        raise ValueError("reduce_sum_triton is a row-wise 2D starter kernel.")

    x = x.contiguous().to(torch.float32)
    rows, cols = x.shape
    if cols > 1024:
        raise ValueError("starter reduction expects at most 1024 columns per row")
    out = torch.empty((rows,), device=x.device, dtype=torch.float32)

    @triton.jit
    def kernel(x_ptr, y_ptr, stride_xm, stride_xn, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        values = tl.load(x_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0)
        total = tl.sum(values, axis=0)
        tl.store(y_ptr + row, total)

    block_size = 1 << (cols - 1).bit_length()
    grid = lambda meta: (rows,)
    kernel[grid](
        x,
        out,
        x.stride(0),
        x.stride(1),
        cols,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


def reduce_sum(x, axis: int = -1, backend: str = "auto"):
    if backend == "numpy":
        return reduce_sum_reference(x, axis=axis)
    if backend == "triton":
        return reduce_sum_triton(x)
    if backend == "auto" and axis == -1 and gpu_stack_ready():
        try:
            import torch
        except Exception:
            pass
        else:
            if torch.is_tensor(x) and x.is_cuda:
                return reduce_sum_triton(x)
    return reduce_sum_reference(x, axis=axis)


if __name__ == "__main__":
    sample = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    print(reduce_sum(sample))
