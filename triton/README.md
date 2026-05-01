# Triton Track

This folder holds Triton notes, design sketches, and lesson support.
Executable Triton code lives in `triton_kernels/` so the real `triton` package
is never shadowed.

## Implementation Files

- `../triton_kernels/vector_add.py`
- `../triton_kernels/softmax.py`
- `../triton_kernels/matmul.py`

## Why Triton

Triton gives students a practical bridge from Python ML workflows to custom GPU
kernels. CUDA remains the mental model. Triton becomes the fast implementation
path for many AI-kernel lessons.

## Suggested Order

1. vector add
2. blocks and masks
3. softmax
4. matmul
5. tuning
6. normalization kernels
7. attention-style kernels
