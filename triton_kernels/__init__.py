"""Optional Triton kernel implementations for GPU machines."""

from .matmul import matmul, matmul_reference, matmul_triton
from .layernorm import layernorm, layernorm_reference, layernorm_triton
from .runtime import (
    as_numpy,
    gpu_stack_ready,
    has_cuda_runtime,
    has_torch,
    has_triton,
    is_torch_cuda_tensor,
    is_torch_cuda_tensor_pair,
)
from .reduce_sum import reduce_sum, reduce_sum_reference, reduce_sum_triton
from .softmax import softmax, softmax_reference, softmax_triton
from .vector_add import vector_add, vector_add_reference, vector_add_triton

__all__ = [
    "as_numpy",
    "gpu_stack_ready",
    "has_cuda_runtime",
    "has_torch",
    "has_triton",
    "is_torch_cuda_tensor",
    "is_torch_cuda_tensor_pair",
    "layernorm",
    "layernorm_reference",
    "layernorm_triton",
    "matmul",
    "matmul_reference",
    "matmul_triton",
    "reduce_sum",
    "reduce_sum_reference",
    "reduce_sum_triton",
    "softmax",
    "softmax_reference",
    "softmax_triton",
    "vector_add",
    "vector_add_reference",
    "vector_add_triton",
]
