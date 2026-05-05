# GPU Kernel Engineer From Scratch

A 12-month CUDA, Triton, and AI systems course where you build a public GPU
kernels portfolio one week at a time.

## Start Here

This repo is the 1-year roadmap to become a GPU kernel engineer.

If you do not watch any video, follow this order:

1. Open [Week 01: GPU Mental Model And Baseline](weeks/week-01-gpu-mental-model.md)
2. Do every task in that file from top to bottom.
3. When Week 01 is done, open [weeks/README.md](weeks/README.md) and continue in order through Week 10.
4. Use the scaffolded weekly files in `weeks/` for the rest of the year.
5. Use [course/syllabus.md](course/syllabus.md) only as the full map.
6. Use [course/recovery-system.md](course/recovery-system.md) if you fall behind.
7. Use [FINISH_PLAN.md](FINISH_PLAN.md) when you want the repo brought to its
   finished public-project state.

Run the starter repo:

```bash
python -m pip install -e ".[dev]"
pytest
python examples/reference_bench.py
make bootstrap-results
make bench
```

## A To B Path

Point A:

- you know Python
- you may use PyTorch
- you do not yet understand GPU kernels deeply
- you do not have a GPU-systems portfolio

Point B:

- you can write CUDA and Triton kernels
- you can test kernels against trusted baselines
- you can benchmark and explain performance
- you can build AI-relevant kernels like softmax, matmul, layer norm, and attention pieces
- you have a public portfolio repo with results, notes, and interview-ready explanations

How you get there:

1. Follow one week file at a time in `weeks/`.
2. Each week, produce one artifact: code, test, benchmark, note, or portfolio section.
3. Each month, use the fourth week to catch up and package your work.
4. By Month 12, turn the artifacts into a final capstone and interview story.

Do not try to speedrun the whole roadmap. The course works because the skills
compound week by week.

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

There are 48 weekly files because 12 months x 4 weeks = 48 weeks.

Month 1: GPU Foundations
- Week 01: GPU mental model and baseline
- Week 02: CUDA setup and vector add
- Week 03: Tensor shapes, memory layout, indexing
- Week 04: Elementwise kernel patterns

Month 2: Memory And Benchmarking
- Week 05: Memory bandwidth and AXPY
- Week 06: Coalescing vs strides
- Week 07: Timing harness and benchmarking
- Week 08: Reading performance results

Month 3: Reductions
- Week 09: Reductions mental model
- Week 10: Naive reduction kernels
- Week 11: Block-level reductions with shared memory
- Week 12: Warp-level reductions

Month 4: Scans, Atomics, Synchronization
- Week 13: Synchronization and barriers
- Week 14: Atomics and contention
- Week 15: Prefix sum and scan mental model
- Week 16: Parallel scan implementation

Month 5: Softmax And Normalization
- Week 17: Softmax math for kernels
- Week 18: Fused row-wise softmax
- Week 19: LayerNorm kernel mental model
- Week 20: RMSNorm kernel

Month 6: Matmul Foundations
- Week 21: Naive matrix multiplication
- Week 22: Tiled matrix multiplication
- Week 23: Tile sizes and occupancy
- Week 24: Matmul checkpoint

Month 7: Triton For AI Kernels
- Week 25: Triton mental model
- Week 26: Triton blocks and masks
- Week 27: Triton softmax
- Week 28: Triton checkpoint

Month 8: Triton Matmul And Tuning
- Week 29: Triton matmul
- Week 30: Autotuning
- Week 31: Batched matmul
- Week 32: Tuning checkpoint

Month 9: PyTorch Integration
- Week 33: PyTorch baselines
- Week 34: Custom op wrapper
- Week 35: GPU test matrix
- Week 36: Integration checkpoint

Month 10: Transformer Kernels
- Week 37: GELU fusion
- Week 38: RMSNorm
- Week 39: Attention pieces
- Week 40: Transformer checkpoint

Month 11: Attention And Inference
- Week 41: Attention forward pass
- Week 42: FlashAttention concepts
- Week 43: KV cache
- Week 44: Attention checkpoint

Month 12: Portfolio And Interviews
- Week 45: Benchmark dashboard
- Week 46: Interview explanations
- Week 47: Resume and story
- Week 48: Final capstone

The detailed week-by-week plan is in [course/syllabus.md](course/syllabus.md),
and the first ten weekly lessons live in [weeks/](weeks/).

## What To Do Each Week

Each rewritten week follows the same shape:

1. Read the current week file.
2. Study the mental model and code-shaped examples.
3. Use the matching `results/` file to capture the main takeaway.
4. Move to the next lesson.

The weekly file is the source of truth. The syllabus tells you where the course
is going, but the weekly file tells you what to do today.

## How The Course Prevents Burnout

- Each lesson focuses on one GPU engineering idea.
- Result notes stay lightweight.
- Later implementation work builds on the lesson files instead of replacing them.
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
- `cuda/` contains standalone CUDA C++ starter kernels and their notes.
- `triton/` contains Triton docs and implementation notes.
- `triton_kernels/` contains executable Triton Python kernels.
- `kernels/` organizes AI-kernel topics independent of implementation language.
- `gputriton/` contains current portable reference implementations.
- `examples/` contains runnable demos.
- `tests/` contains correctness checks.
- `results/` is where benchmark tables and charts should go.
- `portfolio/` contains resume, interview, and project-packaging material.
- `creator/` contains channel cadence, content packaging, and publishing workflow.
- `bonus/10-day-sprint/` contains optional compressed practice material.
- `FINISH_PLAN.md` describes the path from scaffold to finished project.
