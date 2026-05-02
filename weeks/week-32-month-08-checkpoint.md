# Week 32: Month 08 Checkpoint

## What This Week Is

You stop and turn Month 8 into a clean portfolio checkpoint. The goal is to
show that you can explain tuning, matmul, and batching without extra noise, and
that you can say which choice mattered more than the others.

## What To Read

- [../course/month-08-triton-matmul-and-tuning.md](../course/month-08-triton-matmul-and-tuning.md)
- [week-31-batched-matmul.md](week-31-batched-matmul.md)

## Exact Commands

```bash
pytest tests/test_reference.py tests/test_gpu_tracks.py
python examples/reference_bench.py
```

## Build This

Write `results/month-08-checkpoint.md` with the three most important things you
learned in Month 8 and one short portfolio note about tuning or batching.

## Code Sketch

```python
summary = {
    "tiling": "one tile can own a reusable chunk of matmul",
    "tuning": "different settings deserve separate comparison",
    "batching": "the batch dimension repeats the same kernel logic",
}
```

Write one sentence explaining why the sketch is correct before you optimize it.

## Write Down

- What now feels natural?
- What still needs repetition?
- What should Month 9 focus on?
- What is the one matmul idea you would keep in a summary note?

## Minimum

- one checkpoint note
- one portfolio sentence
- one next-step question

## Standard

- compare Month 8 before and after
- add one useful bullet for a README

## Stretch

- add a tiny chart or table
- write one interview-ready answer

## If You Are Behind

Keep the checkpoint short and concrete.

## Next Week

You will move into PyTorch integration and learn how custom ops fit into a
real workflow.
