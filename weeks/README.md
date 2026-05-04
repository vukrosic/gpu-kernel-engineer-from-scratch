# Weekly Course Files

This is the main course path.

Each week should have one Markdown file that can be followed from top to bottom:

- what to learn
- what to run
- what to build
- what to write down
- what counts as done
- what to skip if you are behind

## Available Weeks

- [Week 01: GPU Mental Model And Baseline](week-01-gpu-mental-model.md)
- [Week 02: GPU Setup And Vector Add](week-02-gpu-setup-and-vector-add.md)
- [Week 03: Tensor Shapes, Memory Layout, And Indexing](week-03-grids-blocks-threads-and-indexing.md)
- [Week 04: Elementwise Kernel Patterns](week-04-elementwise-kernel-patterns.md)
- [Week 05: Memory Bandwidth And AXPY](week-05-memory-bandwidth-and-axpy.md)
- [Week 06: Coalescing Vs Strides](week-06-coalescing-vs-strides.md)
- [Week 07: Timing Harness And Benchmarking](week-07-timing-harness-and-benchmarking.md)
- [Week 08: Reading Performance Results](week-08-reading-performance-results.md)
- [Week 09: Naive Reductions](week-09-naive-reductions.md)
- [Week 10: Shared Memory Reductions](week-10-shared-memory-reductions.md)

## How To Use This Folder

Start with the first incomplete week. Do not jump around unless you are reviewing.

For each week:

1. Open the week file.
2. Follow it from top to bottom.
3. Create the requested `results/` note.
4. Stop when the Done Checklist is complete.
5. Move to the next week.

If a week feels too large, do the Minimum checklist. If you have extra time, do
Standard or Stretch.

Every week file is written to be followable top to bottom. Many of the later
weeks are shorter outline lessons, but they still include the same shape:
commands, a build target, a note file, a code sketch, and a next-week preview.

## Year Scaffold

The rest of the year is scaffolded here so the 12-month roadmap is visible at a
glance:

- Month 03: [Week 11](week-11-warp-level-thinking.md), [Week 12](week-12-month-03-checkpoint.md)
- Month 04: [Week 13](week-13-synchronization-and-barriers.md), [Week 14](week-14-atomics-and-histograms.md), [Week 15](week-15-prefix-sum-and-scan.md), [Week 16](week-16-month-04-checkpoint.md)
- Month 05: [Week 17](week-17-softmax-math.md), [Week 18](week-18-fused-softmax.md), [Week 19](week-19-layernorm.md), [Week 20](week-20-month-05-checkpoint.md)
- Month 06: [Week 21](week-21-naive-matmul.md), [Week 22](week-22-tiled-matmul.md), [Week 23](week-23-tiling-and-occupancy.md), [Week 24](week-24-month-06-checkpoint.md)
- Month 07: [Week 25](week-25-triton-mental-model.md), [Week 26](week-26-triton-blocks-and-masks.md), [Week 27](week-27-triton-softmax.md), [Week 28](week-28-month-07-checkpoint.md)
- Month 08: [Week 29](week-29-triton-matmul.md), [Week 30](week-30-autotuning.md), [Week 31](week-31-batched-matmul.md), [Week 32](week-32-month-08-checkpoint.md)
- Month 09: [Week 33](week-33-pytorch-baselines.md), [Week 34](week-34-custom-op-wrapper.md), [Week 35](week-35-gpu-test-matrix.md), [Week 36](week-36-month-09-checkpoint.md)
- Month 10: [Week 37](week-37-gelu-fusion.md), [Week 38](week-38-rmsnorm.md), [Week 39](week-39-attention-pieces.md), [Week 40](week-40-month-10-checkpoint.md)
- Month 11: [Week 41](week-41-attention-forward.md), [Week 42](week-42-flashattention-concepts.md), [Week 43](week-43-kv-cache.md), [Week 44](week-44-month-11-checkpoint.md)
- Month 12: [Week 45](week-45-benchmark-dashboard.md), [Week 46](week-46-interview-explanations.md), [Week 47](week-47-resume-and-story.md), [Week 48](week-48-final-capstone.md)

## Rule

When in doubt, follow the current week file. The syllabus is the map. The week
file is the actual checklist.
