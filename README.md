# GPU Kernel Engineer From Scratch

A 12-month CUDA, Triton, and AI systems course where you build a public GPU
kernels portfolio one week at a time.

This course does not promise a specific salary or job. It promises something
you can control: correct kernels, real benchmarks, clear explanations, and
portfolio artifacts that prove you understand GPU performance engineering.

## Start Here

This repo is the 1-year roadmap to become a GPU kernel engineer.

1. Start the course with [Week 01: GPU Mental Model And Baseline](weeks/week-01-gpu-mental-model.md)
2. Read the full course map: [course/syllabus.md](course/syllabus.md)
3. Read the recovery system: [course/recovery-system.md](course/recovery-system.md)
4. Pick the current assignment: [assignments/README.md](assignments/README.md)
5. Run the starter repo:

```bash
python -m pip install -e ".[dev]"
pytest
python examples/reference_bench.py
```

## Course Promise

Every week, you build one GPU systems skill and ship one portfolio artifact.

By the end, you should be able to demonstrate:

- CUDA kernels, grids, blocks, threads, and warps
- GPU memory hierarchy and performance bottlenecks
- correctness testing against CPU, NumPy, or PyTorch references
- benchmarking, profiling, and performance reports
- reductions, scans, softmax, layer norm, matmul, and attention-style kernels
- CUDA and Triton implementations of AI-relevant operations
- a public repo that can be discussed in ML systems and AI infrastructure interviews

## 12-Month Roadmap

| Month | Theme | Outcome |
| --- | --- | --- |
| 1 | GPU Foundations | Write first correct kernels and explain GPU execution. |
| 2 | Memory And Benchmarking | Measure memory bandwidth and build a reliable benchmark harness. |
| 3 | Reductions | Build row sum, row max, and shared-memory reductions. |
| 4 | Scans, Atomics, Synchronization | Understand coordination between threads. |
| 5 | Softmax And Normalization | Build transformer-adjacent kernels. |
| 6 | Matmul Foundations | Build and tune matrix multiplication. |
| 7 | Triton For AI Kernels | Rebuild key kernels in Triton. |
| 8 | Triton Matmul And Tuning | Tune matmul with block sizes, warps, and stages. |
| 9 | PyTorch Integration | Connect custom kernels to ML workflows. |
| 10 | Transformer Kernels | Build fused MLP, GELU, RMSNorm, and attention pieces. |
| 11 | Attention And Inference | Build simplified attention and KV-cache demos. |
| 12 | Portfolio And Interviews | Package the capstone, benchmarks, and resume story. |

The detailed week-by-week plan is in [course/syllabus.md](course/syllabus.md).

## How The Course Prevents Burnout

- Each month has three build weeks and one checkpoint week.
- Each week has one catch-up day and one lighter rest/portfolio day.
- Every assignment has Minimum, Standard, and Stretch versions.
- If you fall behind, use [course/recovery-system.md](course/recovery-system.md) instead of quitting.

The rule is simple: correct and finished beats perfect and abandoned.

## Repo Structure

- `course/` contains the full 12-month roadmap, weekly rhythm, and recovery system.
- `weeks/` contains one follow-it-top-to-bottom file per course week.
- `assignments/` contains the assignment index and reusable assignment template.
- `cuda/` is the home for CUDA C++ kernels and notes as the course expands.
- `triton/` is the home for Triton kernels and notes as the course expands.
- `kernels/` organizes AI-kernel topics independent of implementation language.
- `gputriton/` contains current portable reference implementations.
- `examples/` contains runnable demos.
- `tests/` contains correctness checks.
- `results/` is where benchmark tables and charts should go.
- `portfolio/` contains resume, interview, and project-packaging material.
- `creator/` contains channel cadence, content packaging, and publishing workflow.
- `bonus/10-day-sprint/` contains optional compressed practice material.

## Channel Positioning

Public promise:

```text
In 12 months, build a public GPU kernels portfolio that proves you understand
CUDA, Triton, performance engineering, and AI kernel optimization.
```

Use salary and career aspiration in titles carefully, but keep the actual course
serious: kernels, benchmarks, explanations, and portfolio proof.
