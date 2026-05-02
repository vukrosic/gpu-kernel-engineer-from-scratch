# Week 40: Month 10 Checkpoint

## What This Week Is

You stop and package the transformer-kernel month. The goal is to turn GELU
fusion, RMSNorm, and attention pieces into a short story you can reuse later.

## What To Read

- [../course/month-10-transformer-kernels.md](../course/month-10-transformer-kernels.md)
- [week-39-attention-pieces.md](week-39-attention-pieces.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write the month 10 checkpoint note and summarize the three most important ideas
from the month. Keep the summary short enough that you could read it out loud
in one minute.

## Code Sketch

```python
month_10_takeaways = [
    ("GELU fusion", "remove one extra write"),
    ("RMSNorm", "one reduction and one scale"),
    ("attention pieces", "score, mask, softmax, value mix"),
]
```

The sketch is correct because it preserves the three lessons in the same order
you would explain them in an interview or README.

Write `results/month-10-checkpoint.md` with the three most important things you
learned in Month 10.

## Write Down

- What is now clearer than before?
- What still needs more repetition?
- What should Month 11 focus on?

## Minimum

- one checkpoint note
- one portfolio summary
- one next-step question

## Standard

- compare Month 10 before and after
- add one interview note

## Stretch

- add a small summary diagram
- write one resume bullet idea

## If You Are Behind

Keep the checkpoint short and useful.

## Next Week

You will begin the attention month and build the forward pass story end to end.
