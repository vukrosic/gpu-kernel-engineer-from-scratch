# 12-Month Syllabus

This is the full GPU Kernels From Scratch roadmap. The early course favors
short lessons that build one concept at a time.

## Month 1: GPU Foundations

Goal: understand the GPU programming model and write the first correct kernels.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 1 | GPU mental model | Explain CPU vs GPU execution and run the starter benchmark. | "Why GPUs are fast for parallel workloads." |
| 2 | CUDA setup and first kernel | Write vector add and compare against a CPU reference. | `vector_add` test and benchmark table. |
| 3 | Tensor shapes, memory layout, and indexing | Understand row-major layout, flattening, strides, and tensor indexing. | Tensor indexing explanation. |
| 4 | Elementwise kernel patterns | Read copy, scale, square, ReLU, add, and axpy-shaped kernels. | Elementwise kernel pattern explanation. |

Minimum viable month: one correct vector add kernel, one test, one benchmark.

## Month 2: Memory And Benchmarking

Goal: learn why memory movement dominates many kernels.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 5 | Memory bandwidth and AXPY | Understand bytes moved, bandwidth, and low arithmetic intensity. | Memory bandwidth explanation. |
| 6 | Coalescing vs strides | Understand contiguous access, strided access, warps, and memory coalescing. | Coalescing explanation. |
| 7 | Timing harness and benchmarking | Understand warmups, repeats, synchronization, median timing, and fair comparisons. | Timing reliability explanation. |
| 8 | Reading performance results | Understand result tables, bandwidth estimates, fair comparisons, and honest conclusions. | Performance interpretation explanation. |

Minimum viable month: one reliable benchmark harness and one bandwidth comparison.

## Month 3: Reductions

Goal: build kernels where many values combine into fewer values.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 9 | Reductions mental model | Understand sum, max, axes, output shape, and coordination. | Reductions explanation. |
| 10 | Naive reduction kernels | Read one-thread-per-output row sum and row max kernels. | Naive reduction explanation. |
| 11 | Block-level reductions with shared memory | Explain how one block cooperates through shared memory to reduce one output region. | Shared-memory reduction explanation. |
| 12 | Warp-level reductions | Understand how warp execution and shuffle operations change reduction design. | Warp-level reduction explanation. |

Quarterly reset: review Months 1-3, fix broken tests, and record a walkthrough.

## Month 4: Scans, Atomics, And Synchronization

Goal: understand coordination between threads.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 13 | Synchronization and barriers | Understand block-local barriers and race conditions. | Synchronization explanation. |
| 14 | Atomics and contention | Understand atomic updates, histograms, and hot counters. | Atomic contention explanation. |
| 15 | Prefix sum and scan mental model | Understand inclusive scan, exclusive scan, and offsets. | Scan mental model explanation. |
| 16 | Parallel scan implementation | Understand staged block scan with shared memory and barriers. | Parallel scan explanation. |

Minimum viable month: one atomic kernel and one clear explanation.

## Month 5: Softmax And Normalization

Goal: build transformer-adjacent kernels people recognize.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 17 | Softmax math for kernels | Understand stable row-wise softmax as max, exp, sum, and divide. | Kernel-shaped softmax explanation. |
| 18 | Fused row-wise softmax | Understand how stable softmax becomes one kernel-shaped pipeline. | Fused softmax explanation. |
| 19 | LayerNorm kernel mental model | Understand row mean, variance, normalization, gamma, and beta. | LayerNorm kernel explanation. |
| 20 | RMSNorm kernel | Understand sum of squares, inverse RMS, and learned weights. | RMSNorm kernel explanation. |

Minimum viable month: correct row-wise softmax and numerical-stability explanation.

## Month 6: Matmul Foundations

Goal: understand the kernel behind most deep learning compute.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 21 | Naive matrix multiplication | Understand one output cell as one dot product across K. | Naive matmul indexing explanation. |
| 22 | Tiled matrix multiplication | Understand shared-memory tiles and reuse across a C tile. | Tiled matmul explanation. |
| 23 | Matmul memory reuse | Understand how A and B tile values are reused across a C tile. | Matmul reuse explanation. |
| 24 | Occupancy, registers, and tile size | Understand how tile choices consume GPU resources. | Occupancy and tile-size explanation. |

Quarterly reset: review Months 4-6 and update resume bullets.

## Month 7: Triton For AI Kernels

Goal: use Triton as a higher-level kernel tool for modern AI workloads.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 25 | Triton mental model | Understand program instances, offsets, and masks. | Triton mental model explanation. |
| 26 | Triton vector add and masks | Understand masked loads and stores on edge blocks. | Triton masks explanation. |
| 27 | Triton reductions | Understand row-wise sum and max reductions in Triton. | Triton reduction explanation. |
| 28 | Triton row-wise softmax | Understand softmax as Triton row ownership plus reductions. | Triton softmax explanation. |

Minimum viable month: one Triton kernel, one test, one benchmark.

## Month 8: Triton Matmul And Kernel Tuning

Goal: learn practical tuning in the style of modern ML infrastructure.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 29 | Triton matmul basics | Understand one program as one C tile with a K loop. | Triton matmul explanation. |
| 30 | Triton matmul performance knobs | Understand block sizes, warps, stages, and shape-dependent tuning. | Tuning explanation. |
| 31 | Batched matmul indexing | Understand batch strides and 3D launch grids. | Batched indexing explanation. |
| 32 | Profiling GPU kernels | Understand timing, profiling, baselines, and bottleneck hypotheses. | Profiling note. |

Minimum viable month: one tuned Triton matmul benchmark.

## Month 9: PyTorch Integration

Goal: connect custom kernels to the ML ecosystem you actually use.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 33 | PyTorch baselines | Compare custom kernels to PyTorch ops. | Baseline benchmark script. |
| 34 | Custom op wrapper | Wrap one kernel in a clean Python API. | Usable package interface. |
| 35 | Testing GPU code | Add shape, dtype, and tolerance tests. | Test matrix for one kernel. |
| 36 | Debugging GPU kernels | Narrow failed GPU tests methodically. | Debugging workflow note. |

Quarterly reset: polish Triton and PyTorch integration work.

## Month 10: Transformer Kernels

Goal: build kernels tied directly to transformer workloads.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 37 | GELU and activation fusion | Fuse bias and activation. | Fused MLP benchmark. |
| 38 | Residual and norm fusion | Understand residual add plus normalization dataflow. | Fusion dataflow note. |
| 39 | Attention scores and masks | Build QK score computation and masking reference. | Attention score walkthrough. |
| 40 | Transformer kernel dataflow | Explain how transformer operations map to kernel patterns. | "Where transformer time goes." |

Minimum viable month: one fused transformer-adjacent kernel and one benchmark.

## Month 11: Attention And Inference

Goal: build a serious capstone around attention and inference bottlenecks.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 41 | Attention forward pass | Implement simplified attention forward. | Correct attention output test. |
| 42 | FlashAttention concepts | Explain tiling and memory reuse. | FlashAttention concept note. |
| 43 | KV cache basics | Build a small KV-cache simulation or benchmark. | Inference bottleneck writeup. |
| 44 | Attention capstone plan | Connect attention forward, FlashAttention concepts, and KV cache into one project story. | Capstone direction. |

Minimum viable month: simplified attention with tests and explanation.

## Month 12: Portfolio, Interviews, And Capstone

Goal: turn the work into a visible career asset.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 45 | Benchmark dashboard | Create final benchmark tables and charts. | `results/` benchmark section. |
| 46 | Interview explanations | Answer memory, matmul, softmax, and attention questions. | Interview question bank. |
| 47 | Resume and project story | Turn the repo into resume bullets. | Resume-ready project section. |
| 48 | Final capstone | Record or write the final project walkthrough. | Public capstone demo. |

Final reset: fix docs, simplify the repo, and plan the next advanced series.
