# Week 21: Naive Matmul

## What This Week Is

You turn two small matrices into a readable baseline matrix multiply. The goal
is correctness first: understand the loops, the indices, and the shape of the
result before any tiling or tuning.

## What To Read

- [../course/month-06-matmul-foundations.md](../course/month-06-matmul-foundations.md)
- [../gputriton/reference.py](../gputriton/reference.py)
- [../cuda/naive_matmul.cu](../cuda/naive_matmul.cu)
- [../triton_kernels/matmul.py](../triton_kernels/matmul.py)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-21-naive-matmul.md` with the shapes you tried, the loop
order you used, and one note about correctness.

## Code Sketch

```python
def matmul_reference(a, b):
    m, k = len(a), len(a[0])
    _, n = len(b), len(b[0])
    out = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total += a[i][p] * b[p][j]
            out[i][j] = total
    return out
```

This sketch is correct because each output cell collects the full dot product
for one row of `a` and one column of `b` before storing the answer.

## Write Down

Answer:

1. How does each output cell get computed?
2. What is slow about the naive version?
3. What would you change next week?
4. What shape checks do you want to keep in mind?

## Minimum

- one readable matmul sketch
- one result note
- one explanation in plain language

## Standard

- compare two shapes
- note one correctness check

## Stretch

- add a tiny timing comparison
- mention one memory traffic issue

## If You Are Behind

Keep the loops small and the note short.

## Next Week

You will tile matmul so the work is grouped into reusable blocks.
