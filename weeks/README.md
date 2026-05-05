# Weekly Course Files

This is the main course path.

Each week should have one Markdown file that can be read from top to bottom:

- what to learn
- the mental model
- small code-shaped examples
- how the idea connects to previous weeks
- how the idea prepares the next week

## Available Weeks

- [Week 01: GPU Mental Model And Baseline](week-01-gpu-mental-model.md)
- [Week 02: GPU Setup And Vector Add](week-02-gpu-setup-and-vector-add.md)
- [Week 03: Tensor Shapes, Memory Layout, And Indexing](week-03-grids-blocks-threads-and-indexing.md)
- [Week 04: Elementwise Kernel Patterns](week-04-elementwise-kernel-patterns.md)
- [Week 05: Memory Bandwidth And AXPY](week-05-memory-bandwidth-and-axpy.md)
- [Week 06: Coalescing Vs Strides](week-06-coalescing-vs-strides.md)
- [Week 07: Timing Harness And Benchmarking](week-07-timing-harness-and-benchmarking.md)
- [Week 08: Reading Performance Results](week-08-reading-performance-results.md)
- [Week 09: Reductions Mental Model](week-09-reductions-mental-model.md)
- [Week 10: Naive Reduction Kernels](week-10-naive-reduction-kernels.md)
- [Week 11: Block-Level Reductions With Shared Memory](week-11-block-level-reductions-with-shared-memory.md)
- [Week 12: Warp-Level Reductions](week-12-warp-level-reductions.md)
- [Week 13: Synchronization And Barriers](week-13-synchronization-and-barriers.md)
- [Week 14: Atomics And Contention](week-14-atomics-and-contention.md)
- [Week 15: Prefix Sum And Scan Mental Model](week-15-prefix-sum-scan-mental-model.md)
- [Week 16: Parallel Scan Implementation](week-16-parallel-scan-implementation.md)
- [Week 17: Softmax Math For Kernels](week-17-softmax-math-for-kernels.md)
- [Week 18: Fused Row-Wise Softmax](week-18-fused-row-wise-softmax.md)
- [Week 19: LayerNorm Kernel Mental Model](week-19-layernorm-kernel-mental-model.md)
- [Week 20: RMSNorm Kernel](week-20-rmsnorm-kernel.md)
- [Week 21: Naive Matrix Multiplication](week-21-naive-matmul.md)
- [Week 22: Tiled Matrix Multiplication](week-22-tiled-matmul.md)
- [Week 23: Matmul Memory Reuse](week-23-matmul-memory-reuse.md)
- [Week 24: Occupancy, Registers, And Tile Size](week-24-occupancy-registers-and-tile-size.md)
- [Week 25: Triton Mental Model](week-25-triton-mental-model.md)
- [Week 26: Triton Vector Add And Masks](week-26-triton-vector-add-and-masks.md)
- [Week 27: Triton Reductions](week-27-triton-reductions.md)
- [Week 28: Triton Row-Wise Softmax](week-28-triton-row-wise-softmax.md)
- [Week 29: Triton Matmul Basics](week-29-triton-matmul-basics.md)
- [Week 30: Triton Matmul Performance Knobs](week-30-triton-matmul-performance-knobs.md)
- [Week 31: Batched Matmul Indexing](week-31-batched-matmul-indexing.md)
- [Week 32: Profiling GPU Kernels](week-32-profiling-gpu-kernels.md)

## How To Use This Folder

Start with the first incomplete week. Do not jump around unless you are reviewing.

For each week:

1. Open the week file.
2. Read it from top to bottom.
3. Use the matching `results/` note to capture the main takeaway.
4. Move to the next week.

The rewritten weeks are lessons first. The matching result files are lightweight
notes for preserving the core idea.

## Year Scaffold

The rest of the year is scaffolded here so the 12-month roadmap is visible at a
glance:

- Month 09: [Week 33](week-33-pytorch-baselines.md), [Week 34](week-34-custom-op-wrapper.md), [Week 35](week-35-gpu-test-matrix.md), [Week 36](week-36-debugging-gpu-kernels.md)
- Month 10: [Week 37](week-37-gelu-fusion.md), [Week 38](week-38-residual-and-norm-fusion.md), [Week 39](week-39-attention-pieces.md), [Week 40](week-40-transformer-kernel-dataflow.md)
- Month 11: [Week 41](week-41-attention-forward.md), [Week 42](week-42-flashattention-concepts.md), [Week 43](week-43-kv-cache.md), [Week 44](week-44-month-11-checkpoint.md)
- Month 12: [Week 45](week-45-benchmark-dashboard.md), [Week 46](week-46-interview-explanations.md), [Week 47](week-47-resume-and-story.md), [Week 48](week-48-final-capstone.md)

## Rule

When in doubt, follow the current week file. The syllabus is the map. The week
file is the actual lesson.
