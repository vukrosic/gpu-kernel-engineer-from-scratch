# Week 06: Coalescing Vs Strides

## What This Week Is

This week turns memory into a shape problem. The work is similar to Week 05, but
now you focus on whether access is contiguous or strided and why that changes
performance behavior.

You are still doing simple operations. The point is to feel that memory layout
is not a detail; it is part of the algorithm.

## What You Need From The Repo

- [../gputriton/bench.py](../gputriton/bench.py)
- [../course/month-02-memory-and-benchmarking.md](../course/month-02-memory-and-benchmarking.md)
- [../weeks/week-05-memory-bandwidth-and-axpy.md](../weeks/week-05-memory-bandwidth-and-axpy.md)

## Exact Commands

```bash
python - <<'PY'
import numpy as np
from gputriton.bench import benchmark

rng = np.random.default_rng(1)
x = rng.normal(size=(2048, 2048))

contiguous = lambda a: np.ascontiguousarray(a)
strided = lambda a: a[:, ::2].copy()

print("contiguous", benchmark(contiguous, x, repeats=50))
print("strided", benchmark(strided, x, repeats=50))
PY
```

Then run:

```bash
pytest
python examples/reference_bench.py
```

## Build This

Create `results/week-06-coalescing-vs-strides.md` and record:

- what contiguous means
- what strided means
- which experiment was faster and why you think that happened
- one sketch of how a GPU would feel the difference

## Code Sketch

```python
def copy_strided(x, step=2):
    out = []
    for i in range(0, len(x), step):
        out.append(x[i])
    return out
```

This sketch is correct because it follows the stride exactly. It also makes the
access pattern visible, which is the point of the week.

## Write Down

Answer these in the note:

1. Why does contiguous access usually make life easier for hardware?
2. What makes a strided pattern harder to handle?
3. Why do memory layout choices matter before you think about optimization?
4. What is the connection between layout and coalescing?

## Minimum

- both experiments run
- the note exists
- you explain contiguous vs strided in your own words

## Standard

- you compare at least one larger and one smaller shape
- you explain why a kernel can be correct and still slow
- you add one sentence about cache or data locality

## Stretch

- you add a third pattern, such as transposed access
- you compare the result to Week 05 bandwidth numbers
- you write a one-paragraph rule of thumb for future kernels

## If You Are Behind

Keep the note simple. The goal is to feel the difference between access
patterns, not to build a benchmark lab.

## Next Week

Week 07 is about timing correctly so you can trust the numbers you record.
