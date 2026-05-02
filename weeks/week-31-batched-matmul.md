# Week 31: Batched Matmul

## What This Week Is

You take the matmul pattern and repeat it across a batch dimension. The goal is
to see how one good kernel idea scales across many small problems without
making the indexing story collapse.

## What To Read

- [../course/month-08-triton-matmul-and-tuning.md](../course/month-08-triton-matmul-and-tuning.md)
- [week-30-autotuning.md](week-30-autotuning.md)
- [../triton_kernels/matmul.py](../triton_kernels/matmul.py)
- [../gputriton/reference.py](../gputriton/reference.py)

## Exact Commands

```bash
pytest tests/test_reference.py tests/test_gpu_tracks.py
python examples/reference_bench.py
```

## Build This

Write `results/week-31-batched-matmul.md` with one batch sketch, one note about
whether batching changes the bottleneck, and one concrete indexing example.

## Code Sketch

```python
def batched_matmul(batch_a, batch_b):
    out = []
    for a, b in zip(batch_a, batch_b):
        out.append(matmul_tile(a, b))
    return out
```

Write one sentence explaining why the sketch is correct before you optimize it.

## Write Down

- What stays the same across the batch?
- What changes for indexing?
- What gets easier to reuse?
- Where does the batch dimension sit relative to the tiles?

## Minimum

- one batch sketch
- one note file
- one short summary

## Standard

- compare two batch sizes
- note one indexing detail

## Stretch

- connect batching to inference workloads
- explain one memory-layout issue

## If You Are Behind

Keep the batch example tiny.

## Next Week

You will package Month 8 and decide what belongs in a portfolio note.
