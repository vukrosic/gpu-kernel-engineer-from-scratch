# Week 44: Month 11 Checkpoint

## What This Week Is

You stop and package the attention month. The goal is to connect the forward
pass, FlashAttention concepts, and KV cache into one inference story.

## What To Read

- [../course/month-11-attention-and-inference.md](../course/month-11-attention-and-inference.md)
- [week-43-kv-cache.md](week-43-kv-cache.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write the month 11 checkpoint note and summarize the three ideas as one flow:
compute attention, reduce memory traffic, then reuse cached keys and values
during decoding.

## Code Sketch

```python
month_11_takeaways = [
    ("attention forward pass", "score, mask, normalize, mix"),
    ("FlashAttention concepts", "tile work to save memory"),
    ("KV cache", "reuse keys and values during decode"),
]
```

The sketch is correct because it preserves the month as a story about inference
bottlenecks, not just three separate notes.

Write `results/month-11-checkpoint.md` with the clearest lessons from Month 11
and one portfolio note.

## Write Down

- What is now easier to explain?
- What still needs repetition?
- What should Month 12 focus on?

## Minimum

- one checkpoint note
- one portfolio summary
- one next-step question

## Standard

- compare Month 11 before and after
- add one interview bullet

## Stretch

- add a tiny summary table
- write one resume bullet idea

## If You Are Behind

Keep the checkpoint short and useful.

## Next Week

You will switch to the final portfolio month and make the work easy to show.
