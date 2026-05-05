# Month 10: Transformer Kernels

## Goal

Build kernels tied directly to transformer workloads.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 37 | GELU and activation fusion | Fuse bias and activation. | Fused operation is benchmarked. |
| 38 | Residual and norm fusion | Understand residual add plus normalization dataflow. | You can explain the temporary tensor fusion avoids. |
| 39 | Attention scores and masks | Understand QK scores, scaling, causal masks, and padding masks. | Attention score walkthrough exists. |
| 40 | Transformer kernel dataflow | Connect matmul, softmax, normalization, residuals, and fusion inside a transformer block. | You can explain where transformer time goes. |

## Minimum Viable Month

One fused transformer-adjacent kernel and one benchmark.

## Portfolio Note

Connect every kernel to a recognizable transformer operation.
