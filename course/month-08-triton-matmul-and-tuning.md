# Month 08: Triton Matmul And Tuning

## Goal

Learn practical kernel tuning in the style of modern ML infrastructure work.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 29 | Triton matmul basics | Understand one program as one C tile with a K loop. | You can explain Triton matmul offsets. |
| 30 | Triton matmul performance knobs | Understand block sizes, warps, stages, and shape-dependent tuning. | You can describe a fair tuning comparison. |
| 31 | Batched matmul indexing | Understand batch strides and 3D launch grids. | You can explain batched matmul indexing. |
| 32 | Profiling GPU kernels | Understand timing, profiling, baselines, and bottleneck hypotheses. | You can write an honest profiling note. |

## Minimum Viable Month

One tuned Triton matmul benchmark.

## Portfolio Note

Show how one tuning choice changed performance and explain why.

## Code Paths

- Triton implementation package: `triton_kernels/matmul.py`
- Reference baseline: `gputriton/reference.py`
