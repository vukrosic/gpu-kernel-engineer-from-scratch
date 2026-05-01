# Month 07: Triton For AI Kernels

## Goal

Use Triton as a practical implementation path for AI kernels.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 25 | Triton mental model | Vector add in Triton. | One Triton kernel is tested. |
| 26 | Blocks and masks | Elementwise or reduction kernels. | Masks are explained. |
| 27 | Triton softmax | Row-wise softmax in Triton. | Benchmark exists. |
| 28 | Checkpoint | CUDA vs Triton comparison. | Short essay is published. |

## Minimum Viable Month

One Triton kernel, one test, and one benchmark.

## Portfolio Note

Explain CUDA as the mental model and Triton as a productive AI-kernel tool.

## Code Paths

- Triton implementation package: `triton_kernels/`
- Reference baseline: `gputriton/reference.py`
