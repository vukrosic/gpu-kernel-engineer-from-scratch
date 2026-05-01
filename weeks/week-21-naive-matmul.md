# Week 21: Naive Matmul

## What This Week Is

You turn two small matrices into a readable baseline matrix multiply. The goal
is correctness first: understand the loops, the indices, and the shape of the
result before any tiling.

## What To Read

- [../course/month-06-matmul-foundations.md](../course/month-06-matmul-foundations.md)
- [week-template.md](week-template.md)
- [../cuda/naive_matmul.cu](../cuda/naive_matmul.cu)
- [../triton_kernels/matmul.py](../triton_kernels/matmul.py)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

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

Write one sentence explaining why the sketch is correct before you optimize it.

Write `results/week-21-naive-matmul.md` with the shapes you tried, the loop
order you used, and one note about correctness.

Use `gputriton/reference.py` as the baseline, then compare the lesson sketch
to `cuda/naive_matmul.cu` and the Triton matmul implementation later in the
course.

## Write Down

- How does each output cell get computed?
- What is slow about the naive version?
- What would you change next week?

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
