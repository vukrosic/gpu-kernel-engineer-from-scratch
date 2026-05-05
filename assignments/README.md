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
| 23 | Matmul tuning | Block-size comparison. |
| 24 | Month 6 checkpoint | Matmul portfolio page. |
| 25 | Triton vector add | One Triton kernel with test. |
| 26 | Triton masks | Elementwise or reduction kernel using masks. |
| 27 | Triton softmax | Row softmax in Triton. |
| 28 | Month 7 checkpoint | CUDA vs Triton comparison. |
| 29 | Triton matmul | Tiled Triton matmul benchmark. |
| 30 | Tuning grid | Compare warps, stages, and block sizes. |
| 31 | Batched matmul | Batched matmul tests. |
| 32 | Month 8 checkpoint | Size vs speed chart. |
| 33 | PyTorch baseline | Compare against PyTorch ops. |
| 34 | Kernel wrapper | Clean Python API for one kernel. |
| 35 | GPU test matrix | Shape, dtype, and tolerance tests. |
| 36 | Month 9 checkpoint | Installation and demo docs. |
| 37 | Fused activation | Bias plus GELU or ReLU fusion. |
| 38 | RMSNorm | RMSNorm comparison with LayerNorm. |
| 39 | Attention pieces | QK scores and masking reference. |
| 40 | Month 10 checkpoint | Transformer bottleneck note. |
| 41 | Attention forward | Simplified attention correctness. |
| 42 | FlashAttention concept | Memory reuse explanation. |
| 43 | KV cache | KV-cache simulation or benchmark. |
| 44 | Month 11 checkpoint | Capstone draft. |
| 45 | Benchmark dashboard | Final tables and charts. |
| 46 | Interview explanations | Kernel interview question bank. |
| 47 | Resume story | 3-5 resume bullets and project narrative. |
| 48 | Final capstone | Public walkthrough and final README polish. |

Use [week-template.md](week-template.md) when creating a detailed assignment file.
