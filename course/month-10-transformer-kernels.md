# Month 10: Transformer Kernels

## Goal

Build kernels tied directly to transformer workloads.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 37 | GELU and activation fusion | Fuse bias and activation. | Fused operation is benchmarked. |
| 38 | RMSNorm | Compare RMSNorm with LayerNorm. | Correctness and tradeoffs are documented. |
| 39 | Attention pieces | QK scores and masking reference. | Attention math walkthrough exists. |
| 40 | Checkpoint | Transformer bottleneck note. | You can explain where transformer time goes. |

## Minimum Viable Month

One fused transformer-adjacent kernel and one benchmark.

## Portfolio Note

Connect every kernel to a recognizable transformer operation.
