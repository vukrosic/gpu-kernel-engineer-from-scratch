# Month 06: Matmul Foundations

## Goal

Understand the kernel behind most deep learning compute.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 21 | Naive matrix multiplication | Understand one output cell as one dot product across K. | You can explain the naive matmul indexing. |
| 22 | Tiled matrix multiplication | Understand shared-memory tiles and reuse across a C tile. | You can explain why tiling reduces repeated global loads. |
| 23 | Matmul memory reuse | Understand how A and B tile values are reused across a C tile. | You can explain why tiling raises arithmetic intensity. |
| 24 | Occupancy, registers, and tile size | Understand how tile choices consume GPU resources. | You can explain why bigger tiles are not automatically better. |

## Minimum Viable Month

One correct matmul kernel and one benchmark against a trusted baseline.

## Portfolio Note

Explain why matmul dominates deep learning and what tiling is trying to reuse.

## Code Paths

- NumPy reference: `gputriton/reference.py`
- CUDA starter: `cuda/naive_matmul.cu`
- Triton starter: `triton_kernels/matmul.py`
