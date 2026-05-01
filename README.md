# GPU Kernel Engineer From Scratch

A 12-month CUDA, Triton, and AI systems course where you build a public GPU
kernels portfolio one week at a time.

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

| Month | Theme | Week 1 | Week 2 | Week 3 | Week 4 |
| --- | --- | --- | --- | --- | --- |
| 1 | GPU Foundations | GPU mental model and baseline | CUDA setup and vector add | Grids, blocks, threads, indexing | Checkpoint and Month 1 writeup |
| 2 | Memory And Benchmarking | Global memory bandwidth | Coalesced vs strided access | Reliable timing harness | Memory bandwidth report |
| 3 | Reductions | Row sum and row max | Shared-memory reductions | Warp-level thinking | Reduction benchmark report |
| 4 | Scans, Atomics, Synchronization | Barriers and race conditions | Atomics and histograms | Prefix sum / scan | Synchronization interview notes |
| 5 | Softmax And Normalization | Safe row-wise softmax | Fused softmax | LayerNorm | Normalization systems note |
| 6 | Matmul Foundations | Naive matmul | Tiled matmul | Tile sizes and occupancy | Matmul portfolio page |
| 7 | Triton For AI Kernels | Triton mental model | Blocks and masks | Triton softmax | CUDA vs Triton comparison |
| 8 | Triton Matmul And Tuning | Triton matmul | Autotuning ideas | Batched matmul | Size vs speed chart |
| 9 | PyTorch Integration | PyTorch baselines | Custom op wrapper | GPU test matrix | Installation and demo docs |
| 10 | Transformer Kernels | GELU and activation fusion | RMSNorm | Attention pieces | Transformer bottleneck note |
| 11 | Attention And Inference | Attention forward pass | FlashAttention concepts | KV cache basics | Capstone draft |
| 12 | Portfolio And Interviews | Benchmark dashboard | Interview explanations | Resume and project story | Final capstone |

The detailed week-by-week plan is in [course/syllabus.md](course/syllabus.md).

## How The Course Prevents Burnout

- Each month has three build weeks and one checkpoint week.
- Each week has one catch-up day and one lighter rest/portfolio day.
- Every assignment has Minimum, Standard, and Stretch versions.
- If you fall behind, use [course/recovery-system.md](course/recovery-system.md) instead of quitting.

The rule is simple: correct and finished beats perfect and abandoned.

## Community

The repo is the free roadmap. The community is for feedback, accountability, and
help finishing the work.

Join here: [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about)

Inside the community, the goal is to help you:

- stay on pace with the weekly roadmap
- ask questions when a kernel, benchmark, or setup step breaks
- get feedback on portfolio notes, benchmark tables, and repo structure
- join office hours and implementation review sessions
- compare your work with other builders following the same path
- turn finished assignments into resume bullets and interview explanations

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
