# Week 01: GPU Mental Model And Baseline

## What This Week Is

This is the first week of the 1-year GPU Kernels From Scratch roadmap.

You are not trying to become fast yet. You are building the habit that will
carry the whole year:

1. understand the concept
2. run the code
3. verify correctness
4. measure something
5. write down what happened

This week uses CPU/NumPy reference code on purpose. Before writing CUDA kernels,
you need trusted baselines that future GPU kernels can be tested against.

## Outcome

By the end of this week, you will have:

- installed and run the project
- run the correctness tests
- run the starter benchmark
- written a short CPU vs GPU mental model
- created your first benchmark note in `results/`
- understood what Week 02 will build next

## Time Budget

Minimum path: 2-3 hours total.

Standard path: 4-6 hours total.

Stretch path: 6-8 hours total.

If you are busy, do the Minimum path. Staying in motion matters more than making
Week 01 perfect.

## Day 1: Read The Course Promise

Read these files:

- [../README.md](../README.md)
- [../course/recovery-system.md](../course/recovery-system.md)
- [../course/month-01-gpu-foundations.md](../course/month-01-gpu-foundations.md)

Then write this in your own words in a notebook or `results/week-01-baseline.md`:

```text
I am taking this course to build a public GPU kernels portfolio.
This year is about correct kernels, benchmarks, and clear explanations.
The goal is not to memorize CUDA. The goal is to become the kind of engineer who
can reason about GPU performance.
```

Now write your own version. Keep it short. Five sentences is enough.

## Day 2: Set Up The Repo

From the repo root, run:

```bash
python -m pip install -e ".[dev]"
```

Then run:

```bash
pytest
```

Expected result:

```text
4 passed
```

If tests fail, do not continue. Fix setup first.

If you are using a machine without a GPU, that is fine for Week 01. This week is
about the mental model and reference baselines.

## Day 3: Run The Starter Benchmark

Run:

```bash
python examples/reference_bench.py
```

You should see timings for:

- `vector_add`
- `matmul`
- `softmax`
- `attention`

The exact numbers do not matter yet. Your job is to record them.

Create a file:

```text
results/week-01-baseline.md
```

Add this:

```text
# Week 01 Baseline

Machine:
Python version:
Command:

## Results

vector_add:
matmul:
softmax:
attention:

## Notes

What surprised me:
What I do not understand yet:
What I want to compare when GPU kernels exist:
```

Fill it in from your run.

## Day 4: Understand The Reference Code

Open:

- [../gputriton/reference.py](../gputriton/reference.py)
- [../gputriton/bench.py](../gputriton/bench.py)
- [../tests/test_reference.py](../tests/test_reference.py)

Read the functions in this order:

1. `vector_add`
2. `softmax`
3. `matmul`
4. `attention`
5. `benchmark`
6. `run_reference_benchmarks`

Write one sentence for each function:

```text
vector_add:
softmax:
matmul:
attention:
benchmark:
run_reference_benchmarks:
```

Do not overthink it. The goal is to know what the baseline code does.

## Day 5: Learn The GPU Mental Model

Write this down:

```text
A CPU is optimized for complex sequential control flow.
A GPU is optimized for doing many similar operations in parallel.
A CUDA kernel is a function that runs many times across many GPU threads.
Threads are grouped into blocks.
Blocks are grouped into a grid.
The hard part is not only writing the math. The hard part is moving memory and
organizing parallel work efficiently.
```

Now rewrite it in your own words.

Then answer these questions:

1. Why is vector add a good first GPU kernel?
2. Why is matmul more important for AI than vector add?
3. Why do we need a CPU or NumPy reference before writing a GPU kernel?
4. What does a benchmark tell us that a correctness test does not?
5. What could go wrong if we optimize before checking correctness?

Put your answers in `results/week-01-baseline.md`.

## Day 6: Catch Up Or Clean Up

If you are behind, do only this:

1. Run `pytest`.
2. Run `python examples/reference_bench.py`.
3. Write five sentences explaining CPU vs GPU.

That is the Minimum assignment.

If you are on track, clean up `results/week-01-baseline.md` so another person
could read it.

## Day 7: Portfolio Note And Rest

Add this section to `results/week-01-baseline.md`:

```text
## Portfolio Note

This week I set up the GPU Kernels From Scratch repo, ran the reference tests,
and recorded baseline timings for vector add, matmul, softmax, and attention.
The most important idea is that future GPU kernels need trusted CPU/NumPy
baselines so I can prove correctness before optimizing performance.
```

Rewrite that in your own voice.

Then stop. Do not start Week 02 early unless you are genuinely excited and have
extra energy. The course is one year long. Recovery is part of the system.

## Done Checklist

Minimum:

- [ ] `pytest` passes
- [ ] `python examples/reference_bench.py` runs
- [ ] You wrote five sentences explaining CPU vs GPU

Standard:

- [ ] `results/week-01-baseline.md` exists
- [ ] It includes benchmark numbers
- [ ] It includes one-sentence explanations of the reference functions
- [ ] It includes answers to the five mental-model questions

Stretch:

- [ ] You ran the benchmark three times and compared variation
- [ ] You changed the input sizes in `gputriton/bench.py` and observed what changed
- [ ] You wrote one question you want Week 02 to answer

## What To Skip If Overwhelmed

Skip Stretch.

Skip changing benchmark input sizes.

Do not skip tests. Do not skip the baseline benchmark. Those are the habits the
whole course depends on.

## What Week 02 Will Do

Week 02 starts the first real GPU kernel path:

- CUDA setup
- vector add as the first kernel
- CPU/NumPy reference comparison
- first GPU benchmark

The Week 01 baseline is what lets you know whether the Week 02 kernel is correct.
