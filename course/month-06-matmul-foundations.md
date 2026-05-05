# Month 06: Matmul Foundations

## Goal

Understand the kernel behind most deep learning compute.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 21 | Naive matrix multiplication | Understand one output cell as one dot product across K. | You can explain the naive matmul indexing. |
| 22 | Tiled matrix multiplication | Understand shared-memory tiles and reuse across a C tile. | You can explain why tiling reduces repeated global loads. |
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
