# Week 29: Triton Matmul

## What This Week Is

You map matrix multiplication onto Triton blocks. The goal is to see how the
same math from Month 6 becomes a kernel-shaped implementation strategy.

## What To Read

- [../course/month-08-triton-matmul-and-tuning.md](../course/month-08-triton-matmul-and-tuning.md)
- [week-24-month-06-checkpoint.md](week-24-month-06-checkpoint.md)
- [../triton_kernels/matmul.py](../triton_kernels/matmul.py)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

## Code Sketch

```python
def matmul_tile(a_tile, b_tile):
    acc = [[0.0 for _ in range(len(b_tile[0]))] for _ in range(len(a_tile))]
    for i in range(len(a_tile)):
        for j in range(len(b_tile[0])):
            for k in range(len(a_tile[0])):
                acc[i][j] += a_tile[i][k] * b_tile[k][j]
    return acc
```

Write one sentence explaining why the sketch is correct before you optimize it.

Write `results/week-29-triton-matmul.md` with a tile sketch and one note on
how Triton expresses the same idea as the earlier matmul weeks.

Use `gputriton/reference.py` and `cuda/naive_matmul.cu` as the comparison
points for the Triton version.

## Write Down

- What is shared between CUDA and Triton matmul thinking?
- What changes in the code shape?
- What should be benchmarked later?

## Minimum

- one Triton matmul sketch
- one note file
- one plain-language summary

## Standard

- compare two tile choices
- note one reuse or masking detail

## Stretch

- sketch a batched matmul variant
- explain one performance tradeoff

## If You Are Behind

Keep the sketch small and the note short.

## Next Week

You will learn how tuning changes the shape of a Triton kernel search.
