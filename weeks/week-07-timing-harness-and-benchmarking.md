# Week 07: Timing Harness And Benchmarking

## What This Week Is

Week 07 turns benchmarking into a habit instead of a guess. You build a tiny
timing harness, learn how to repeat measurements, and see why a single number is
usually not enough.

## What You Need From The Repo

- [../gputriton/bench.py](../gputriton/bench.py)
- [../course/month-02-memory-and-benchmarking.md](../course/month-02-memory-and-benchmarking.md)
- [../weeks/week-05-memory-bandwidth-and-axpy.md](../weeks/week-05-memory-bandwidth-and-axpy.md)
- [../weeks/week-06-coalescing-vs-strides.md](../weeks/week-06-coalescing-vs-strides.md)

## Exact Commands

```bash
python - <<'PY'
import statistics
import numpy as np
from gputriton.bench import benchmark

rng = np.random.default_rng(2)
a = rng.normal(size=500_000)
b = rng.normal(size=500_000)

samples = [benchmark(lambda x, y: 1.5 * x + y, a, b, repeats=25) for _ in range(7)]
print("samples", samples)
print("median", statistics.median(samples))
print("mean", statistics.mean(samples))
PY
```

Run the repo checks again:

```bash
pytest
python examples/reference_bench.py
```

## Build This

Create `results/week-07-timing-harness.md` and include:

- your own simple timing harness
- the difference between one measurement and repeated measurements
- why warmup matters
- why median can be more useful than one raw number

Also compare your measurements from Week 05 and Week 06 if you can.

## Code Sketch

```python
def repeated_measure(run, repeats=7):
    samples = [run() for _ in range(repeats)]
    return sorted(samples)[len(samples) // 2]
```

This sketch is correct because it runs the same action several times and uses a
middle value instead of trusting one noisy result.

## Write Down

Answer these in the note:

1. Why is one timing result not enough?
2. Why do repeats help?
3. What is the point of warmup?
4. When would you trust the median more than the mean?

## Minimum

- you run the sample harness
- `results/week-07-timing-harness.md` exists
- you explain warmup and repeats in simple language

## Standard

- you compare median and mean
- you run the harness on two different inputs
- you write one note about variability

## Stretch

- you build a reusable helper function for repeated timing
- you add a simple table of results in your note
- you explain how bad timing habits can create fake optimization wins

## If You Are Behind

Use the sample harness and keep your note short. You do not need to turn this
into a profiling project yet.

## Next Week

Week 08 is a checkpoint for Month 2, where you package the memory story and
your benchmark habits into a clean summary.
