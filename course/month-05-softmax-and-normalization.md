# Month 05: Softmax And Normalization

## Goal

Build transformer-adjacent kernels that appear in real AI systems.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 17 | Softmax math for kernels | Understand stable row-wise softmax as max, exp, sum, and divide. | You can explain the kernel-shaped softmax pipeline. |
| 18 | Fused row-wise softmax | Understand how stable softmax becomes one kernel-shaped pipeline. | You can explain fusion as a memory-traffic optimization. |
| 19 | LayerNorm kernel mental model | Understand row mean, variance, normalization, gamma, and beta. | You can describe the LayerNorm kernel pipeline. |
| 20 | RMSNorm kernel | Understand sum of squares, inverse RMS, and learned weights. | You can explain what RMSNorm removes from LayerNorm. |

## Minimum Viable Month

Correct row-wise softmax and a numerical-stability explanation.

## Portfolio Note

Connect softmax and normalization to transformer workloads.

## Code Paths

- NumPy reference: `gputriton/reference.py`
- CUDA starter: `cuda/softmax.cu`
- Triton starter: `triton_kernels/softmax.py`
- CUDA starter: `cuda/layernorm.cu`
- Triton starter: `triton_kernels/layernorm.py`
