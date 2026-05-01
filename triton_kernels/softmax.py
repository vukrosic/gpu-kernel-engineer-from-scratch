from __future__ import annotations

import numpy as np

from .runtime import as_numpy, gpu_stack_ready, is_torch_cuda_tensor, require_gpu_stack


def softmax_reference(x, axis: int = -1):
    x = as_numpy(x).astype(np.float64, copy=False)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def softmax_triton(x):
    torch, triton, tl = require_gpu_stack()
    if not torch.is_tensor(x):
        raise TypeError("softmax_triton expects a torch tensor.")
    if not x.is_cuda:
        raise ValueError("softmax_triton expects a CUDA tensor.")
    if x.ndim != 2:
        raise ValueError("softmax_triton is a row-wise 2D starter kernel.")

    x = x.contiguous().to(torch.float32)
    rows, cols = x.shape
    if cols > 1024:
        raise ValueError("starter softmax expects at most 1024 columns per row")
    out = torch.empty_like(x)

    @triton.jit
    def kernel(x_ptr, y_ptr, stride_xm, stride_xn, stride_ym, stride_yn, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x_vals = tl.load(x_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=-float("inf"))
        row_max = tl.max(x_vals, axis=0)
        shifted = x_vals - row_max
        exp_vals = tl.exp(shifted)
        denom = tl.sum(exp_vals, axis=0)
        probs = exp_vals / denom
        tl.store(y_ptr + row * stride_ym + cols * stride_yn, probs, mask=mask)

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
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


def softmax(x, axis: int = -1, backend: str = "auto"):
    if backend == "numpy":
        return softmax_reference(x, axis=axis)
    if backend == "triton":
        return softmax_triton(x)
    if backend == "auto" and axis == -1 and is_torch_cuda_tensor(x) and gpu_stack_ready():
        return softmax_triton(x)
    return softmax_reference(x, axis=axis)


if __name__ == "__main__":
    sample = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    print(softmax(sample))
