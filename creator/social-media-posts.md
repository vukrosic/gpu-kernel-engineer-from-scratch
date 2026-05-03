# Social Media Posts

## Title

GPU Kernels From Scratch: The Roadmap For ML Engineers

## Post

If you want to go from "I use PyTorch" to "I understand what is happening underneath PyTorch," this is the roadmap.

The goal is not to memorize CUDA syntax.

The goal is to build the skill stack behind modern AI systems:

- understand how GPU programs actually execute
- write simple CUDA kernels from scratch
- test kernels against trusted baselines
- benchmark performance without fooling yourself
- understand memory bandwidth, coalescing, synchronization, and tiling
- build AI-relevant kernels like softmax, LayerNorm, matmul, and attention pieces
- use Triton as a practical tool for writing ML kernels
- explain your work clearly enough for portfolio and interview use

Here is what you need to learn.

The necessary path is about 12 serious build weeks. Some people will take longer, some people will move faster, but the order matters because each topic depends on the previous one.

Every week should produce three things:

- one correct thing
- one measured thing
- one explained thing

That is the whole game.

Inside Skool, some of the beginning steps will run as weekly challenges where members can submit their work and get it checked.

The first challenge is:

Build and explain your first `vector_add` GPU kernel.

That challenge covers setup, indexing, bounds checks, correctness testing, and the first benchmark.

## Week 1: GPU Mental Model And Setup

Learn:

- what a kernel is
- how CPU execution differs from GPU execution
- what grids, blocks, threads, and warps mean
- how to run the project and baseline tests

Tasks:

1. Set up the repo and run the starter tests.
2. Run the reference benchmark.
3. Write a short explanation of CPU serial execution vs GPU parallel execution.
4. Draw or describe how one vector is split across many GPU threads.
5. Record your first baseline result, even if it is CPU-only.

Questions to answer:

- What work is each thread responsible for?
- Why is vector add a good first GPU kernel?
- What does the GPU do well that the CPU does not?
- What part of the setup confused you?

Ship:

A short baseline note explaining your current mental model.

## Week 2: First CUDA Kernel

Learn:

- CUDA file structure
- kernel launch syntax
- device memory allocation
- copying data between host and device
- indexing with `blockIdx`, `blockDim`, and `threadIdx`

Tasks:

1. Implement vector add in CUDA.
2. Compare the output against a CPU or NumPy reference.
3. Test at least five input sizes, including a size that is not divisible by the block size.
4. Benchmark CPU vs CUDA.
5. Write down the indexing formula and explain it in plain English.

Questions to answer:

- What happens when there are more threads than elements?
- Why do we need a bounds check?
- Which input size makes the GPU look better or worse?
- What is the smallest bug you hit while writing the kernel?

Ship:

A correct vector add kernel, tests, and a small benchmark table.

## Week 3: Elementwise Kernels And Indexing Practice

Learn:

- how the same indexing pattern applies across simple kernels
- how to structure small correctness tests
- how to think about shapes, sizes, and edge cases

Tasks:

1. Implement add, multiply, square, ReLU, and scale kernels.
2. Add tests against CPU or NumPy references.
3. Test normal, tiny, empty, and odd-size inputs.
4. Benchmark at least two kernels across multiple sizes.
5. Refactor only if the repeated pattern is obvious.

Questions to answer:

- Which parts of these kernels are identical?
- Which edge case is easiest to forget?
- Why are elementwise kernels often memory-bound?
- What would break if your indexing formula were off by one?

Ship:

An elementwise kernel suite with tests.

## Week 4: Memory Bandwidth And Benchmarking

Learn:

- why memory movement often dominates GPU performance
- what coalesced memory access means
- why warmups and repeated measurements matter
- how to report bandwidth in GB/s

Tasks:

1. Build copy, scale, and AXPY-style kernels.
2. Benchmark each kernel with warmup iterations.
3. Compare coalesced access with strided access.
4. Calculate approximate memory bandwidth.
5. Write a short benchmark report.

Questions to answer:

- How much data does each kernel read and write?
- Why can a mathematically simple kernel still be slow?
- What changed when memory access became strided?
- How would a bad timing method mislead you?

Ship:

A memory bandwidth report with at least one benchmark table.

## Week 5: Reductions

Learn:

- how many values become one value
- why reductions are harder than elementwise kernels
- how shared memory helps within a block
- what synchronization does

Tasks:

1. Implement row sum or row max naively.
2. Add tests across multiple row and column sizes.
3. Implement a shared-memory version.
4. Compare naive vs shared-memory performance.
5. Write a short explanation of the reduction pattern.

Questions to answer:

- Why can each output require many input values?
- Where do threads need to cooperate?
- What does `__syncthreads()` protect?
- When does the optimized version become worth it?

Ship:

A reduction kernel with correctness tests and before/after benchmarks.

## Week 6: Synchronization, Atomics, And Scan

Learn:

- race conditions
- barriers
- atomics
- prefix sum / scan as a coordination pattern

Tasks:

1. Create or study a small race-condition example.
2. Implement a histogram or counting kernel using atomics.
3. Implement a small block-level prefix sum.
4. Test both kernels against CPU references.
5. Write a bug diary: what could go wrong and how you know it is fixed.

Questions to answer:

- What makes a race condition nondeterministic?
- When are atomics useful?
- Why can atomics become slow?
- How is scan different from reduction?

Ship:

One atomic kernel, one scan experiment, and a synchronization note.

## Week 7: Softmax And LayerNorm

Learn:

- numerical stability
- row-wise operations
- why softmax and normalization matter in transformers
- how to compare against PyTorch

Tasks:

1. Implement safe row-wise softmax.
2. Compare against PyTorch or NumPy.
3. Implement a fused softmax path if possible.
4. Implement LayerNorm forward.
5. Benchmark softmax and LayerNorm on several matrix shapes.

Questions to answer:

- Why do we subtract the row max in softmax?
- Which dimensions are reduced in softmax?
- What does LayerNorm compute per row?
- Where are the memory reads and writes?

Ship:

Correct softmax and LayerNorm kernels with tests and a numerical stability explanation.

## Week 8: Matmul Foundations

Learn:

- why matmul dominates deep learning compute
- naive matrix multiplication
- tiled matrix multiplication
- shared memory reuse
- tile size tradeoffs

Tasks:

1. Implement naive matmul.
2. Test across square and rectangular matrices.
3. Implement tiled matmul using shared memory.
4. Benchmark naive vs tiled.
5. Compare your result to a PyTorch baseline.

Questions to answer:

- How many operations does matmul perform?
- Which values get reused?
- Why does tiling help?
- Why is beating PyTorch or cuBLAS not the goal at this stage?

Ship:

Naive and tiled matmul with correctness tests and a benchmark table.

## Week 9: Triton Basics

Learn:

- Triton's programming model
- programs, blocks, masks, and offsets
- how Triton differs from CUDA
- why Triton is useful for ML kernels

Tasks:

1. Rewrite vector add in Triton.
2. Implement at least one elementwise Triton kernel.
3. Use masks for odd-size inputs.
4. Benchmark CUDA vs Triton vs PyTorch if available.
5. Write a CUDA vs Triton comparison.

Questions to answer:

- What does one Triton program operate on?
- Why are masks important?
- What feels easier in Triton than CUDA?
- What control do you lose or hide when using Triton?

Ship:

A small Triton kernel suite with tests and a comparison note.

## Week 10: Triton Softmax And Matmul

Learn:

- row-wise Triton kernels
- Triton matmul structure
- block size choices
- basic autotuning ideas

Tasks:

1. Implement Triton softmax.
2. Test it against PyTorch.
3. Implement Triton matmul or adapt a starter version.
4. Try at least three block configurations.
5. Record which configuration wins for which shapes.

Questions to answer:

- How does Triton express a row-wise operation?
- Which matmul shapes are easiest or hardest?
- What changed when you changed block size?
- What would a real autotuner search over?

Ship:

Triton softmax, Triton matmul, and a tuning table.

## Week 11: Transformer Kernels And Attention Pieces

Learn:

- GELU or activation fusion
- RMSNorm vs LayerNorm
- attention score computation
- masking
- KV-cache basics

Tasks:

1. Implement bias plus GELU or another fused activation.
2. Implement RMSNorm and compare with LayerNorm.
3. Build a reference attention score computation.
4. Add masking behavior and tests.
5. Create a tiny KV-cache simulation or explanation.

Questions to answer:

- Why fuse activation with another operation?
- How is RMSNorm simpler than LayerNorm?
- What does attention compute before softmax?
- Why does KV cache matter for inference?

Ship:

At least one transformer-adjacent kernel plus an attention-piece explanation.

## Week 12: Portfolio, Benchmarks, And Interview Story

Learn:

- how to package technical work
- how to make benchmark results readable
- how to explain tradeoffs without overclaiming
- how to turn the project into interview material

Tasks:

1. Build a benchmark dashboard or final benchmark table.
2. Write a README section that explains the project from start to finish.
3. Pick three kernels and write interview-style explanations.
4. Create resume bullets from the project.
5. Record or write a final walkthrough.

Questions to answer:

- What is the most impressive correct thing you built?
- Which benchmark result taught you the most?
- What limitation should you openly admit?
- How would you explain this project to an ML infrastructure interviewer?

Ship:

A polished portfolio package: README, benchmarks, interview explanations, and resume bullets.

## Recovery Rule

If you fall behind, do not restart from zero.

Shrink the current week to:

1. run one command
2. make one small thing correct
3. measure one result
4. write one sentence about what happened

Then move forward.

Never optimize broken code.

## What To Post In The Community

Each week, post:

- what you built
- the correctness result
- the benchmark result
- what confused you
- the question you want help with

Good community posts are specific.

Bad post:

"I do not understand CUDA."

Better post:

"My vector add works for size 1024 but fails for size 1000. I think my bounds check is wrong. Here is the indexing formula I used."

The roadmap is free.

The community is for finishing it, debugging it, benchmarking it, polishing it, and learning to explain the work like an AI systems engineer.

Start with Week 1.

Run the baseline.

Write down what happened.

Then keep going.
