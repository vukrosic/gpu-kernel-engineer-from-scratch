# Assignments

Every assignment has three levels so you can stay in motion.

| Level | Requirement |
| --- | --- |
| Minimum | Make one thing correct and explain it in one paragraph. |
| Standard | Add tests, benchmarks, and a short portfolio note. |
| Stretch | Optimize, compare variants, and explain tradeoffs. |

## Weekly Assignment Index

| Week | Assignment | Minimum Deliverable |
| --- | --- | --- |
| 1 | [GPU mental model](../weeks/week-01-gpu-mental-model.md) | 5-sentence CPU vs GPU explanation. |
| 2 | Vector add | Correct vector add and one benchmark. |
| 3 | Elementwise suite | Add, multiply, square, and ReLU references or kernels. |
| 4 | Month 1 checkpoint | Clean notes and one benchmark table. |
| 5 | Memory bandwidth | Copy, scale, and axpy benchmark. |
| 6 | Coalescing | Coalesced vs strided access comparison. |
| 7 | Timing harness | Warmup, repeats, median timing. |
| 8 | Month 2 checkpoint | Memory bandwidth writeup. |
| 9 | Row reductions | Row sum and row max correctness. |
| 10 | Shared reductions | One optimized reduction variant. |
| 11 | Block-level reductions with shared memory | Shared-memory reduction explanation. |
| 12 | Warp-level reductions | Warp shuffle reduction explanation. |
| 13 | Synchronization and barriers | Barrier and race-condition explanation. |
| 14 | Atomics and contention | Atomic update and contention explanation. |
| 15 | Prefix sum and scan mental model | Inclusive vs exclusive scan explanation. |
| 16 | Parallel scan implementation | Staged block scan explanation. |
| 17 | Softmax math for kernels | Numerically stable softmax explanation. |
| 18 | Fused row-wise softmax | Fused softmax pipeline explanation. |
| 19 | LayerNorm kernel mental model | LayerNorm row-statistics explanation. |
| 20 | RMSNorm kernel | RMSNorm vs LayerNorm explanation. |
| 21 | Naive matrix multiplication | Naive matmul indexing explanation. |
| 22 | Tiled matrix multiplication | Tiled matmul reuse explanation. |
| 23 | Matmul memory reuse | Matmul reuse explanation. |
| 24 | Occupancy, registers, and tile size | Tile-size tradeoff explanation. |
| 25 | Triton mental model | Triton program/block/mask explanation. |
| 26 | Triton vector add and masks | Edge-block mask explanation. |
| 27 | Triton reductions | Triton sum/max reduction explanation. |
| 28 | Triton row-wise softmax | Triton softmax pipeline explanation. |
| 29 | Triton matmul basics | Triton C-tile indexing explanation. |
| 30 | Triton matmul performance knobs | Tuning knob explanation. |
| 31 | Batched matmul indexing | Batch stride explanation. |
| 32 | Profiling GPU kernels | Profiling and benchmark hygiene note. |
| 33 | PyTorch baseline | Compare against PyTorch ops. |
| 34 | Kernel wrapper | Clean Python API for one kernel. |
| 35 | GPU test matrix | Shape, dtype, and tolerance tests. |
| 36 | Debugging GPU kernels | Methodical debugging workflow. |
| 37 | Fused activation | Bias plus GELU or ReLU fusion. |
| 38 | Residual and norm fusion | Residual add plus normalization dataflow. |
| 39 | Attention scores and masks | QK scores and masking reference. |
| 40 | Transformer kernel dataflow | Transformer bottleneck map. |
| 41 | Attention forward | Simplified attention correctness. |
| 42 | FlashAttention concept | Memory reuse explanation. |
| 43 | KV cache | KV-cache simulation or benchmark. |
| 44 | Month 11 checkpoint | Capstone draft. |
| 45 | Benchmark dashboard | Final tables and charts. |
| 46 | Interview explanations | Kernel interview question bank. |
| 47 | Resume story | 3-5 resume bullets and project narrative. |
| 48 | Final capstone | Public walkthrough and final README polish. |

Use [week-template.md](week-template.md) when creating a detailed assignment file.
