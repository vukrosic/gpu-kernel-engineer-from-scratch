# Portfolio Rubric

A strong GPU kernels portfolio is not just code. It is proof that you can think
like a performance engineer.

## Required Evidence

- Correctness tests against trusted baselines.
- Benchmark tables with hardware and shape details.
- Notes explaining bottlenecks and tradeoffs.
- At least one before/after optimization story.
- At least one AI-relevant kernel: softmax, layer norm, matmul, or attention.

## Portfolio Levels

| Level | Evidence |
| --- | --- |
| Beginner | Correct vector add, elementwise kernels, and benchmarks. |
| Intermediate | Reductions, softmax, matmul, and tuning notes. |
| Advanced | Triton kernels, PyTorch integration, attention or inference capstone. |

## Resume Bullet Pattern

Use this pattern:

```text
Built and benchmarked [kernel/system] in [CUDA/Triton], validating correctness
against [baseline] and improving [metric] by [result] on [hardware].
```

Do not invent numbers. If the benchmark is not impressive yet, describe the
engineering process honestly.
