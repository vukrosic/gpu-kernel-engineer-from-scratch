# Month 07: Triton For AI Kernels

## Goal

Use Triton as a practical implementation path for AI kernels.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 25 | Triton mental model | Understand program instances, offsets, and masks. | You can explain Triton blocks of data. |
| 26 | Triton vector add and masks | Understand masked loads and stores on edge blocks. | You can explain why masks protect correctness. |
| 27 | Triton reductions | Understand row-wise sum and max reductions in Triton. | You can explain reduction identities for masked values. |
| 28 | Triton row-wise softmax | Understand softmax as Triton row ownership plus reductions. | You can describe the fused Triton softmax pipeline. |

## Minimum Viable Month

One Triton kernel, one test, and one benchmark.

## Portfolio Note

Explain CUDA as the mental model and Triton as a productive AI-kernel tool.

## Code Paths

- Triton implementation package: `triton_kernels/`
- Reference baseline: `gputriton/reference.py`
