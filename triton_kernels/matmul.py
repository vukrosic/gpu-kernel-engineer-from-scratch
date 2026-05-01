from __future__ import annotations

import numpy as np

from .runtime import as_numpy, gpu_stack_ready, is_torch_cuda_tensor_pair, require_gpu_stack


def matmul_reference(a, b):
    return as_numpy(a) @ as_numpy(b)


def matmul_triton(a, b):
    torch, triton, tl = require_gpu_stack()
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        raise TypeError("matmul_triton expects torch tensors.")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("matmul_triton expects CUDA tensors.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul_triton is a 2D starter kernel.")

    a = a.contiguous().to(torch.float32)
    b = b.contiguous().to(torch.float32)
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("matmul_triton requires compatible shapes.")

    out = torch.empty((m, n), device=a.device, dtype=torch.float32)

    @triton.jit
    def kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block in range(0, tl.cdiv(K, BLOCK_K)):
            k_start = k_block * BLOCK_K
            k_mask = k_start + offs_k < K
            a_mask = (offs_m < M)[:, None] & k_mask[None, :]
            b_mask = k_mask[:, None] & (offs_n < N)[None, :]
            a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a_vals, b_vals)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=c_mask)

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=16,
        num_warps=4,
        num_stages=2,
    )
    return out


def matmul(a, b, backend: str = "auto"):
    if backend == "numpy":
        return matmul_reference(a, b)
    if backend == "triton":
        return matmul_triton(a, b)
    if backend == "auto" and gpu_stack_ready() and is_torch_cuda_tensor_pair(a, b):
        return matmul_triton(a, b)
    return matmul_reference(a, b)


if __name__ == "__main__":
    sample_a = np.arange(6, dtype=np.float32).reshape(2, 3)
    sample_b = np.arange(12, dtype=np.float32).reshape(3, 4)
    print(matmul(sample_a, sample_b))
