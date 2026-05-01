# GPU Kernels From Scratch Completion Plan

This plan turns the current scaffold into a finished public portfolio project.

The repo already has the main course map, 48 weekly lesson files, starter CUDA
and Triton kernels, lightweight tests, portfolio folders, and creator material.
The missing work is proof: complete implementations, real correctness coverage,
benchmarks, result writeups, and final packaging.

## Definition Of Finished

The project is finished when a reviewer can clone the repo and see:

1. A clear 12-month learning path with all 48 weeks usable from top to bottom.
2. Correct CPU/NumPy references for every major operation.
3. CUDA implementations for the core low-level kernels.
4. Triton implementations for the AI-facing kernels.
5. PyTorch baseline comparisons where relevant.
6. Tests that run on CPU-only machines and expanded GPU tests when CUDA exists.
7. Benchmark scripts that produce reproducible tables.
8. `results/` writeups for every week and stronger monthly checkpoint reports.
9. A final capstone around attention or transformer kernels.
10. Resume bullets, interview explanations, and a polished README.

## Current State

Already present:

- 48 week files in `weeks/`.
- Course map in `course/`.
- Starter CUDA kernels in `cuda/`.
- Starter Triton kernels in `triton_kernels/`.
- CPU reference code in `gputriton/`.
- Basic tests in `tests/`.
- Portfolio and creator folders.

Known gaps:

- `results/` only contains `README.md`.
- Tests are currently lightweight and mostly shape/import checks.
- CUDA code exists, but there is no build/test harness for compiling and running it.
- Triton kernels cover only starter paths and limited shapes.
- Weeks 33-48 need stronger implementation backing for PyTorch integration,
  transformer kernels, attention, KV cache, and capstone.
- Portfolio files are short placeholders rather than final assets.
- Benchmarks do not yet create durable result tables or charts.

## Phase 0: Stabilize The Repo

Goal: make the project easy to run before adding more work.

Tasks:

- Add a `Makefile` or `justfile` with `test`, `bench`, `lint`, and optional
  `cuda-test` commands.
- Add a `scripts/` folder for benchmark and report generation.
- Add a `benchmarks/` folder separate from examples.
- Add clear CPU-only and GPU-enabled setup instructions.
- Expand `pyproject.toml` optional dependencies:
  - `dev`: pytest, ruff or equivalent
  - `gpu`: torch, triton, matplotlib or pandas if charts are generated
- Add environment detection docs so CPU-only users understand skipped GPU paths.

Acceptance checks:

- `python -m pip install -e ".[dev]"` works.
- `pytest` passes on CPU-only.
- A reviewer can tell which commands require CUDA.

## Phase 1: Complete Reference Layer

Goal: every kernel topic has a trusted CPU/NumPy reference.

Implement or expand references for:

- vector add
- elementwise add, multiply, square, ReLU
- copy, scale, AXPY
- coalesced versus strided memory access simulation
- row sum, row max
- histogram/counting reference
- prefix sum/scan reference
- safe softmax
- fused softmax reference
- LayerNorm
- RMSNorm
- matmul
- batched matmul
- bias plus GELU/ReLU fusion
- attention score/masking pieces
- simplified attention forward
- KV-cache simulation

Acceptance checks:

- Every reference has tests across normal, edge, and odd-size inputs.
- Numerical tests use tolerances appropriate to the operation.
- Reference functions are documented enough for students to trust them.

## Phase 2: Finish CUDA Track

Goal: make the low-level CUDA path credible and runnable.

Core CUDA deliverables:

- `vector_add.cu`
- elementwise suite
- copy, scale, AXPY
- row sum and row max reductions
- shared-memory reduction
- histogram or atomic counter
- block-level scan
- safe softmax
- fused softmax
- LayerNorm
- naive matmul
- tiled matmul

Support work:

- Add a CUDA compile script that uses `nvcc` when available.
- Add pytest wrappers that skip cleanly when `nvcc` or CUDA is absent.
- Store representative stdout and benchmark numbers in `results/`.
- Add short CUDA notes for grid/block choices and memory movement.

Acceptance checks:

- CPU-only test suite still passes.
- CUDA tests compile and run on a CUDA machine.
- Each CUDA kernel has a correctness check against the reference layer.
- Month 1-6 result pages are backed by actual code paths.

## Phase 3: Finish Triton Track

Goal: make the modern AI-kernel path complete enough for a portfolio review.

Core Triton deliverables:

- vector add
- elementwise suite
- row sum and row max
- safe softmax
- fused softmax
- LayerNorm
- RMSNorm
- matmul
- tuned matmul variants
- batched matmul
- bias plus activation fusion
- attention pieces
- simplified attention forward

Support work:

- Expand `triton_kernels/` APIs with consistent `backend="auto"` behavior.
- Add shape, dtype, contiguous, non-contiguous, and odd-size tests.
- Add GPU tests that skip when CUDA/Triton is unavailable.
- Add benchmark scripts comparing NumPy, PyTorch, CUDA where available, and Triton.
- Document known limitations such as maximum block size or dtype support.

Acceptance checks:

- Triton GPU tests pass on a CUDA/Triton machine.
- CPU fallback paths pass everywhere.
- Month 7-11 result pages include real benchmark tables.

## Phase 4: PyTorch Integration

Goal: connect the kernels to workflows an ML systems reviewer recognizes.

Tasks:

- Add PyTorch baseline benchmarks for:
  - vector add
  - reductions
  - softmax
  - LayerNorm/RMSNorm
  - matmul/batched matmul
  - attention pieces
- Add a clean public Python API for one or more kernels.
- Add wrapper examples that accept torch tensors.
- Add test matrix coverage:
  - shapes
  - dtypes
  - devices
  - tolerances
  - error cases
- Add docs explaining when the custom kernel is educational versus practical.

Acceptance checks:

- `tests/` separates CPU, optional GPU, and optional integration coverage.
- Benchmarks include PyTorch as the main production baseline.
- README makes the educational scope honest.

## Phase 5: Results And Portfolio Evidence

Goal: make the repo look finished without requiring the reader to run everything.

Create result files:

- `results/week-01-baseline.md` through `results/week-48-final-capstone.md`
- `results/month-01-checkpoint.md` through `results/month-12-checkpoint.md`
- `results/benchmark-dashboard.md`
- `results/gpu-test-matrix.md`
- optional generated charts under `results/figures/`

Each weekly result should include:

- what was built
- correctness check
- benchmark or measured observation
- one lesson learned
- known limitation or next improvement

Each monthly checkpoint should include:

- summary table
- best benchmark
- hardest bug
- interview explanation
- portfolio artifact link

Acceptance checks:

- Every week links to a result file.
- Every result file links back to the relevant source, test, or benchmark.
- Final README points to the best 5-8 artifacts.

## Phase 6: Capstone

Goal: finish with one strong story instead of 48 disconnected exercises.

Recommended capstone:

Build and explain a simplified transformer attention path:

1. PyTorch reference attention.
2. NumPy reference attention.
3. Triton attention pieces.
4. KV-cache simulation.
5. Benchmark against PyTorch on small, medium, and odd sizes.
6. Explain why FlashAttention exists without claiming to fully reimplement it.

Capstone files:

- `capstone/README.md`
- `capstone/attention_reference.py`
- `capstone/attention_triton.py`
- `capstone/kv_cache.py`
- `benchmarks/attention_bench.py`
- `results/week-48-final-capstone.md`

Acceptance checks:

- Capstone has tests.
- Capstone has benchmark output.
- Capstone README explains the system clearly in under five minutes of reading.
- Portfolio material points to the capstone as the final proof.

## Phase 7: Final Polish

Goal: make the project public-facing.

Tasks:

- Rewrite the root README around the finished artifacts, not just the roadmap.
- Add a quickstart section for CPU-only and GPU users.
- Add a project status table.
- Add links to best results, capstone, resume bullets, and interview questions.
- Expand `portfolio/resume-bullets.md`.
- Expand `portfolio/interview-questions.md`.
- Expand `portfolio/portfolio-rubric.md` with final self-assessment.
- Remove generated caches from the repo if tracked.
- Add or verify `.gitignore` coverage for caches, build outputs, benchmark logs,
  and large generated artifacts.

Acceptance checks:

- Fresh clone instructions work.
- `pytest` passes.
- Optional GPU test command is documented.
- The repo tells a coherent story from beginner GPU kernels to transformer
  kernel capstone.

## Suggested Execution Order

1. Phase 0: repo commands and docs.
2. Phase 1: reference layer and CPU tests.
3. Phase 5 partial: create result-file templates for all 48 weeks.
4. Phase 2: CUDA Months 1-6.
5. Phase 3: Triton Months 7-11.
6. Phase 4: PyTorch integration and benchmark comparisons.
7. Phase 6: capstone.
8. Phase 5 completion: fill all results with real outputs.
9. Phase 7: final public polish.

## Practical Completion Milestones

Milestone 1: Runnable scaffold

- Commands are standardized.
- CPU tests pass.
- GPU tests skip cleanly.

Milestone 2: CPU truth layer

- All reference functions exist.
- All reference tests pass.

Milestone 3: CUDA portfolio

- Months 1-6 have compileable CUDA code, tests, and result writeups.

Milestone 4: Triton portfolio

- Months 7-11 have Triton code, tests, benchmarks, and result writeups.

Milestone 5: Capstone

- Attention/KV-cache capstone is tested, benchmarked, and explained.

Milestone 6: Public launch

- README, portfolio files, benchmark dashboard, and final walkthrough are polished.

## Minimum Finished Version

If time gets tight, the minimum credible finish is:

- Complete reference layer.
- Complete tests for references.
- CUDA: vector add, reductions, softmax, LayerNorm, matmul.
- Triton: vector add, softmax, LayerNorm, matmul, attention pieces.
- PyTorch benchmarks for the same operations.
- Result writeups for all checkpoint weeks and major build weeks.
- Final capstone README and benchmark table.
- Polished README and resume bullets.

This is enough to support the claim: "I built and benchmarked a GPU kernels
portfolio from first principles through transformer-adjacent kernels."
