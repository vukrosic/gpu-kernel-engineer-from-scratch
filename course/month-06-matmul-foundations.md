# Month 06: Matmul Foundations

## Goal

Understand the kernel behind most deep learning compute.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 21 | Naive matmul | Basic matrix multiplication. | Correct across matrix sizes. |
| 22 | Tiled matmul | Shared-memory tiled matmul. | TFLOPS benchmark exists. |
| 23 | Tile sizes and occupancy | Compare block sizes and resource usage. | Tuning report is written. |
| 24 | Checkpoint | Matmul portfolio page. | Results and tradeoffs are clear. |

## Minimum Viable Month

One correct matmul kernel and one benchmark against a trusted baseline.

## Portfolio Note

Explain why matmul dominates deep learning and what tiling is trying to reuse.

## Code Paths

- NumPy reference: `gputriton/reference.py`
- CUDA starter: `cuda/naive_matmul.cu`
- Triton starter: `triton_kernels/matmul.py`
