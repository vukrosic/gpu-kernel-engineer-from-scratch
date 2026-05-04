# Month 02: Memory And Benchmarking

## Goal

Learn why memory movement dominates many GPU kernels and how to measure it.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 5 | Memory bandwidth and AXPY | Understand bytes moved, bandwidth, and low arithmetic intensity. | You can explain why simple elementwise kernels are often memory-bound. |
| 6 | Coalescing vs strides | Understand contiguous access, strided access, warps, and memory coalescing. | You can explain why nearby threads should access nearby memory. |
| 7 | Timing correctly | Warmup, repeats, median timing. | Benchmark harness is reusable. |
| 8 | Checkpoint | Memory bandwidth report. | Results are documented. |

## Minimum Viable Month

One reliable benchmark harness and one memory bandwidth comparison.

## Portfolio Note

Write what changed between coalesced and strided access and why the result matters.
