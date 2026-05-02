# Week 05: Memory Bandwidth And Axpy

## What This Week Is

Month 2 begins with the real bottleneck theme: memory movement. Before talking
about fancy kernels, you learn to look at copy, scale, and axpy-style work and
ask how much data moves.

This week is where "faster math" stops being the whole story. On real hardware,
the amount of data you move can matter more than the number of floating-point
operations you do.

## What You Need From The Repo

- [../gputriton/bench.py](../gputriton/bench.py)
- [../gputriton/reference.py](../gputriton/reference.py)
- [../course/month-02-memory-and-benchmarking.md](../course/month-02-memory-and-benchmarking.md)

## Exact Commands

Run the repo checks:

```bash
pytest
python examples/reference_bench.py
```

Run a small bandwidth experiment:

```bash
python - <<'PY'
import numpy as np
from gputriton.bench import benchmark

rng = np.random.default_rng(0)
a = rng.normal(size=1_000_000)
b = rng.normal(size=1_000_000)

print("copy", benchmark(lambda x: np.array(x, copy=True), a, repeats=100))
print("scale", benchmark(lambda x: x * 1.5, a, repeats=100))
print("axpy", benchmark(lambda x, y: 1.5 * x + y, a, b, repeats=100))
PY
```

## Build This

Create `results/week-05-memory-bandwidth.md` and record:

- the copy timing
- the scale timing
- the axpy timing
- what you think changed between them
- what "memory bandwidth" means in plain language

Then write a short note that answers:

- why copy is a useful baseline
- why axpy is a good bridge between math and memory
- why this week matters before coalescing and matmul

## Code Sketch

```python
def axpy(x, y, alpha=1.5):
    return [alpha * xi + yi for xi, yi in zip(x, y)]
```

This sketch is correct because each output element uses exactly one element
from `x` and one from `y`, which makes the memory cost easy to reason about.

## Write Down

Answer these in the note:

1. Why does moving data matter so much on GPUs?
2. Why are copy, scale, and axpy good first memory experiments?
3. What part of the work is arithmetic and what part is memory traffic?
4. What would you expect to change if the arrays were much larger?

## Minimum

- the bandwidth experiment runs
- `results/week-05-memory-bandwidth.md` exists
- you can explain copy, scale, and axpy in one sentence each

## Standard

- you compare at least two input sizes
- you write one paragraph about why memory traffic dominates many kernels
- you note one reason a benchmark can mislead you

## Stretch

- you compute a rough bytes-moved estimate for each experiment
- you compare the results to the vector-add benchmark from Week 02
- you explain why "more math" is not always the same as "more cost"

## If You Are Behind

Do the copy and axpy experiments only. Skip the extra size comparisons, but do
not skip the writeup.

## Next Week

Week 06 compares contiguous and strided access so you can feel the difference
between friendly memory layout and unfriendly memory layout.
