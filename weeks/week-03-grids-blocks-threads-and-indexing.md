# Week 03: Grids, Blocks, Threads, And Indexing

## What This Week Is

Week 03 teaches you how to think about work assignment. The math is still tiny,
but the indexing model becomes the star: how many workers, how they map to data,
and how elementwise kernels stay correct.

This is where the course stops feeling like "just arrays" and starts feeling
like a parallel system. If the index math is wrong, the kernel is wrong even if
the arithmetic is perfect.

## What You Need From The Repo

- [../gputriton/reference.py](../gputriton/reference.py)
- [../course/month-01-gpu-foundations.md](../course/month-01-gpu-foundations.md)
- [../weeks/week-02-gpu-setup-and-vector-add.md](../weeks/week-02-gpu-setup-and-vector-add.md)

## Exact Commands

Explore row-major layout and slicing:

```bash
python - <<'PY'
import numpy as np

x = np.arange(12).reshape(3, 4)
print(x)
print(x.ravel())
print(x[:, 1])
print(x[::2])
PY
```

Run the repo checks again:

```bash
pytest
python examples/reference_bench.py
```

## Build This

Create `results/week-03-indexing.md` and write:

- how a 2D array is laid out in memory
- what "one worker per element" means
- how add, multiply, square, and ReLU fit the same indexing pattern
- how the idea of blocks and threads changes the way you think about a kernel

Write a small mapping table in the note:

```text
data index -> worker id -> action
```

Then do the same idea for a 2D grid and a row-major matrix.

## Code Sketch

```python
def index_2d(row, col, width):
    return row * width + col
```

This sketch is correct because row-major layout stores whole rows contiguously,
so the row offset comes before the column offset.

## Write Down

Answer these in your note:

1. Why is indexing a correctness problem, not just a performance problem?
2. What breaks if two workers write to the same output element?
3. Why do elementwise kernels feel simple but still matter?
4. How does a batch dimension change the way you think about a tensor?

## Minimum

- `results/week-03-indexing.md` exists
- you explain row-major indexing in plain language
- you can map at least one elementwise operation to a worker rule

## Standard

- you compare a 1D vector and a 2D matrix
- you write a short pseudocode sketch for a ReLU kernel
- you explain why one worker per element is easy to reason about

## Stretch

- you sketch how a grid/block/thread hierarchy would cover a 2D tensor
- you explain why memory layout matters before optimization starts
- you draft a tiny checklist for catching indexing bugs

## If You Are Behind

Use only the Minimum path. If you are stuck, go back to Week 02 and reread the
vector-add mapping before moving on.

## Next Week

Week 04 is the Month 1 checkpoint: you package what you have learned into a
clean summary and a portfolio note.
