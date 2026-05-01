# Month 08: Triton Matmul And Tuning

## Goal

Learn practical kernel tuning in the style of modern ML infrastructure work.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 29 | Triton matmul | Tiled matmul in Triton. | Benchmark table exists. |
| 30 | Autotuning ideas | Compare block sizes, warps, stages. | Tuning grid is recorded. |
| 31 | Batched matmul | Extend matmul to batches. | Tests pass across batch sizes. |
| 32 | Checkpoint | Size vs speed chart. | Results are readable. |

## Minimum Viable Month

One tuned Triton matmul benchmark.

## Portfolio Note

Show how one tuning choice changed performance and explain why.
