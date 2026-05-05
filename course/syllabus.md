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
| 12 | Warp-level thinking | Understand how warp execution changes reduction design. | Warp-level reduction explanation. |

Quarterly reset: review Months 1-3, fix broken tests, and record a walkthrough.

## Month 4: Scans, Atomics, And Synchronization

Goal: understand coordination between threads.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 13 | Synchronization | Use barriers safely inside a block. | Race-condition bug diary. |
| 14 | Atomics | Build a histogram or counting kernel. | Atomic kernel with tests. |
| 15 | Prefix sum / scan | Implement a small block-level scan. | Scan explanation and benchmark. |
| 16 | Checkpoint | Compare reductions, atomics, and scans. | Synchronization interview notes. |

Minimum viable month: one atomic kernel and one clear explanation.

## Month 5: Softmax And Normalization

Goal: build transformer-adjacent kernels people recognize.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 17 | Softmax math | Implement safe row-wise softmax from a reference. | Softmax test against PyTorch or NumPy. |
| 18 | Fused softmax | Combine max, exp, sum, and divide in one kernel. | Fused softmax benchmark. |
| 19 | LayerNorm | Implement forward LayerNorm. | LayerNorm correctness and timing. |
| 20 | Checkpoint | Explain why normalization appears in transformers. | Kernel-to-transformers note. |

Minimum viable month: correct row-wise softmax and numerical-stability explanation.

## Month 6: Matmul Foundations

Goal: understand the kernel behind most deep learning compute.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 21 | Naive matmul | Implement basic matrix multiplication. | Tests across matrix sizes. |
| 22 | Tiled matmul | Use shared memory tiling. | TFLOPS benchmark table. |
| 23 | Tile sizes and occupancy | Compare block sizes and resource usage. | Tuning report. |
| 24 | Checkpoint | Clean up matmul code and notes. | Month 6 matmul page. |

Quarterly reset: review Months 4-6 and update resume bullets.

## Month 7: Triton For AI Kernels

Goal: use Triton as a higher-level kernel tool for modern AI workloads.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 25 | Triton mental model | Rewrite vector add in Triton. | CUDA vs Triton comparison. |
| 26 | Triton blocks and masks | Build elementwise and reduction kernels. | Triton kernel suite. |
| 27 | Triton softmax | Implement row-wise softmax in Triton. | Triton softmax benchmark. |
| 28 | Checkpoint | Explain when Triton is useful. | CUDA vs Triton essay. |

Minimum viable month: one Triton kernel, one test, one benchmark.

## Month 8: Triton Matmul And Kernel Tuning

Goal: learn practical tuning in the style of modern ML infrastructure.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 29 | Triton matmul | Build tiled matmul in Triton. | Matmul benchmark table. |
| 30 | Autotuning ideas | Compare block sizes, warps, and stages. | Tuning grid results. |
| 31 | Batched matmul | Extend matmul to batches. | Batched matmul tests. |
| 32 | Checkpoint | Package matmul results clearly. | Size vs speed chart. |

Minimum viable month: one tuned Triton matmul benchmark.

## Month 9: PyTorch Integration

Goal: connect custom kernels to the ML ecosystem you actually use.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 33 | PyTorch baselines | Compare custom kernels to PyTorch ops. | Baseline benchmark script. |
| 34 | Custom op wrapper | Wrap one kernel in a clean Python API. | Usable package interface. |
| 35 | Testing GPU code | Add shape, dtype, and tolerance tests. | Test matrix for one kernel. |
| 36 | Checkpoint | Make the repo easier for reviewers to run. | Installation and demo docs. |

Quarterly reset: polish Triton and PyTorch integration work.

## Month 10: Transformer Kernels

Goal: build kernels tied directly to transformer workloads.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 37 | GELU and activation fusion | Fuse bias and activation. | Fused MLP benchmark. |
| 38 | RMSNorm | Implement RMSNorm and compare with LayerNorm. | Normalization comparison note. |
| 39 | Attention pieces | Build QK score computation and masking reference. | Attention math walkthrough. |
| 40 | Checkpoint | Explain transformer bottlenecks. | "Where transformer time goes." |

Minimum viable month: one fused transformer-adjacent kernel and one benchmark.

## Month 11: Attention And Inference

Goal: build a serious capstone around attention and inference bottlenecks.

| Week | Topic | Assignment | Portfolio Artifact |
| --- | --- | --- | --- |
| 41 | Attention forward pass | Implement simplified attention forward. | Correct attention output test. |
| 42 | FlashAttention concepts | Explain tiling and memory reuse. | FlashAttention concept note. |
| 43 | KV cache basics | Build a small KV-cache simulation or benchmark. | Inference bottleneck writeup. |
| 44 | Checkpoint | Polish attention code and explanations. | Capstone draft. |

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
