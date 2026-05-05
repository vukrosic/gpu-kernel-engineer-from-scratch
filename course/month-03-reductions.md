# Month 03: Reductions

## Goal

Build kernels where many values combine into fewer values.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 9 | Reductions mental model | Understand sum, max, axes, output shape, and coordination. | You can explain why reductions are not elementwise kernels. |
| 10 | Naive reduction kernels | Read one-thread-per-output row sum and row max kernels. | You can explain why the naive kernel is correct but limited. |
| 11 | Block-level reductions with shared memory | Explain how one block cooperates through shared memory to reduce one output region. | You can describe the shared-memory reduction pattern. |
| 12 | Warp-level reductions | Understand how warp execution and shuffle operations change reduction design. | You have a warp-level reduction explanation. |

## Minimum Viable Month

Row sum and row max with tests.

## Portfolio Note

Explain why reductions need coordination and why naive parallelism is not enough.

## Code Paths

- NumPy reference: `gputriton/reference.py`
- CUDA starter: `cuda/reduce_sum.cu`
- Triton starter: `triton_kernels/reduce_sum.py`
