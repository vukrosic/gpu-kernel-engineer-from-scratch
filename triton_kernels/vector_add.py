from __future__ import annotations

import numpy as np

from .runtime import as_numpy, gpu_stack_ready, is_torch_cuda_tensor_pair, require_gpu_stack


def vector_add_reference(a, b):
    return as_numpy(a) + as_numpy(b)


def vector_add_triton(a, b):
    torch, triton, tl = require_gpu_stack()
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        raise TypeError("vector_add_triton expects torch tensors.")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("vector_add_triton expects CUDA tensors.")

    a = a.contiguous().to(torch.float32)
    b = b.contiguous().to(torch.float32)
    out = torch.empty_like(a)
    n = out.numel()

    @triton.jit
    def kernel(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, a_vals + b_vals, mask=mask)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    kernel[grid](a, b, out, n, BLOCK_SIZE=1024, num_warps=4)
    return out


def vector_add(a, b, backend: str = "auto"):
    if backend == "numpy":
        return vector_add_reference(a, b)
    if backend == "triton":
        return vector_add_triton(a, b)
    if backend == "auto" and gpu_stack_ready() and is_torch_cuda_tensor_pair(a, b):
        return vector_add_triton(a, b)
    return vector_add_reference(a, b)


if __name__ == "__main__":
    sample_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sample_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    print(vector_add(sample_a, sample_b))
