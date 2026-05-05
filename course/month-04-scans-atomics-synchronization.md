# Month 04: Scans, Atomics, And Synchronization

## Goal

Understand coordination between threads.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 13 | Synchronization and barriers | Understand block-local barriers and race conditions. | You can explain when `__syncthreads()` is required. |
| 14 | Atomics and contention | Understand atomic updates, histograms, and hot counters. | You can explain why atomics are correct but can be slow. |
| 15 | Prefix sum and scan mental model | Understand inclusive scan, exclusive scan, and offsets. | You can explain why scan is not a reduction. |
| 16 | Parallel scan implementation | Understand staged block scan with shared memory and barriers. | You can describe a block-level scan implementation. |

## Minimum Viable Month

One atomic kernel and one clear explanation of when atomics help or hurt.

## Portfolio Note

Write a short bug diary about a synchronization mistake and how you fixed it.
