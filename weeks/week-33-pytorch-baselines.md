# Week 33: PyTorch Baselines

## What This Week Is

You learn to trust a clean reference before you touch a custom op. The point is
to know what correct looks like, how to compare against it, and why a baseline
is more than just a backup plan.

## What To Read

- [../course/month-09-pytorch-integration.md](../course/month-09-pytorch-integration.md)
- [week-32-month-08-checkpoint.md](week-32-month-08-checkpoint.md)
- [../gputriton/reference.py](../gputriton/reference.py)
- [../tests/test_reference.py](../tests/test_reference.py)

## Exact Commands

```bash
pytest tests/test_reference.py tests/test_gpu_tracks.py
python examples/reference_bench.py
```

## Build This

Write `results/week-33-pytorch-baselines.md` with one baseline table, one note
about why references make debugging easier, and one sentence about what the
baseline protects you from.

## Code Sketch

```python
def compare_baseline(name, reference, candidate, *inputs):
    ref = reference(*inputs)
    out = candidate(*inputs)
    return name, ref.shape == out.shape
```

Write one sentence explaining why the sketch is correct before you optimize it.

## Write Down

- What makes a baseline trustworthy?
- What should be compared against it?
- What is the simplest correctness check?
- Which output property matters first: shape, dtype, or values?

## Minimum

- one baseline note
- one comparison table
- one short explanation

## Standard

- compare two baseline cases
- note one bug the baseline could catch

## Stretch

- sketch a tiny test matrix
- explain one tradeoff in testing

## If You Are Behind

Keep the baseline small and readable.

## Next Week

You will wrap a custom op so the baseline can meet your code at the boundary.
