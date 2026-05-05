# Week 47: Resume And Project Story

Week 47 teaches how to turn the repo into a project story.

This is not separate from engineering.

Clear project writing shows that you understand what you built.

## Step 1: Start With The Reader

A reviewer wants to answer four questions quickly:

```text
what did you build?
how did you verify it?
what performance idea did you explore?
why does it matter for ML engineering?
```

Your project story should make those answers easy to find.

Do not make the reader reconstruct the project from 48 week files.

## Step 2: Write The One-Sentence Project Summary

A good project summary names the domain, method, and proof.

Example:

```text
Built a from-scratch GPU kernel learning repo covering CUDA/Triton kernels,
reference-first correctness checks, benchmark notes, and transformer-focused
topics such as fusion, attention, FlashAttention concepts, and KV cache.
```

That sentence is specific.

It says:

```text
domain: GPU kernels
methods: CUDA/Triton, references, benchmarks
ML relevance: transformers, attention, KV cache
```

## Step 3: Avoid Empty Resume Language

Weak bullet:

```text
Worked on GPU kernels and improved performance.
```

Stronger bullet:

```text
Built reference-first CUDA/Triton kernel lessons and validation notes covering
elementwise ops, reductions, softmax, normalization, matmul, attention, and KV
cache inference patterns.
```

The stronger bullet names the work.

It does not claim a speedup without evidence.

## Step 4: Use Evidence-Based Bullets

A resume bullet should connect action to proof.

Useful proof includes:

```text
tests
benchmarks
profiling notes
shape coverage
baseline comparisons
clear documentation
```

Example:

```text
Designed GPU-kernel correctness workflow using PyTorch baselines, shape/dtype
checks, tolerance-based value comparisons, and targeted test matrices before
benchmarking custom paths.
```

This bullet is credible because it describes a real engineering process.

## Step 5: Include One Performance Story

A performance bullet should be careful:

```text
Explored memory-traffic reductions through tiled matmul, activation fusion,
residual/norm fusion, and FlashAttention-style attention dataflow.
```

This says what was explored without inventing unsupported numbers.

If you have real benchmark numbers, include them.

If not, keep the claim qualitative and precise.

## Step 6: Keep The Story In Order

The project story should follow the learning arc:

```text
1. baseline correctness
2. simple kernels
3. memory access and reductions
4. matmul and tiling
5. Triton kernels
6. PyTorch integration
7. transformer and attention kernels
8. portfolio evidence
```

This order matters because it shows progression.

It also helps a reviewer trust that the project was built deliberately.

## Step 7: Link To Artifacts

A story is stronger when it points to files.

Good links include:

```text
README
weeks directory
results directory
benchmark dashboard
attention capstone note
test files
```

Do not link everything.

Pick the few files that prove the project fastest.

## Example Resume Section

```text
GPU Kernel Engineering From Scratch
- Built a CUDA/Triton learning repo covering elementwise kernels, reductions,
  softmax, normalization, matmul, Triton programming, PyTorch integration, and
  transformer attention dataflow.
- Designed a reference-first validation workflow using PyTorch baselines,
  shape/dtype/device checks, tolerance-based comparisons, and targeted GPU test
  matrices.
- Documented performance reasoning around memory coalescing, shared-memory
  reuse, tiled matmul, fusion, FlashAttention concepts, and KV-cache inference
  tradeoffs.
```

These bullets are useful because they are specific without pretending the repo
is a production kernel library.

## The Core Pattern

A strong project story says:

```text
what was built
which engineering habits made it trustworthy
which performance ideas were studied
which artifacts prove the work
what the next deeper version would improve
```

That is enough for a resume, GitHub profile, or interview intro.

## Bridge To Week 48

Week 48 packages the final capstone.

The final lesson is about making the repo easy to open, understand, verify, and
continue.
