# Week 29: Triton Matmul

## What This Week Is

You map matrix multiplication onto Triton blocks. The point is to see how the
same math from Month 6 becomes a tile-shaped implementation strategy, and how
the mental model now includes both indexing and reuse.

## What To Read

- [../course/month-08-triton-matmul-and-tuning.md](../course/month-08-triton-matmul-and-tuning.md)
- [week-24-month-06-checkpoint.md](week-24-month-06-checkpoint.md)
- [../triton_kernels/matmul.py](../triton_kernels/matmul.py)
- [../gputriton/reference.py](../gputriton/reference.py)
- [../cuda/naive_matmul.cu](../cuda/naive_matmul.cu)

## Exact Commands

```bash
pytest tests/test_reference.py tests/test_gpu_tracks.py
python examples/reference_bench.py
```

## Build This

Write `results/week-29-triton-matmul.md` with one tile sketch, one note about
how Triton expresses the same idea as the earlier matmul weeks, and one
comparison point to the CUDA starter.

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

## Write Down

- What is shared between CUDA and Triton matmul thinking?
- What changes in the code shape?
- What should be benchmarked later?
- Which part of the tile deserves the most reuse?

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
