# Month 02: Memory And Benchmarking

## Goal

Learn why memory movement dominates many GPU kernels and how to measure it.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 5 | Global memory | Copy, scale, and axpy-style kernels. | You can report GB/s. |
| 6 | Memory coalescing | Coalesced vs strided memory access. | You have a performance comparison. |
| 7 | Timing correctly | Warmup, repeats, median timing. | Benchmark harness is reusable. |
| 8 | Checkpoint | Memory bandwidth report. | Results are documented. |

## Minimum Viable Month

One reliable benchmark harness and one memory bandwidth comparison.

## Portfolio Note

Write what changed between coalesced and strided access and why the result matters.
