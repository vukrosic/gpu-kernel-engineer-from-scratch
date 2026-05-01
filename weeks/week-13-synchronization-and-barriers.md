# Week 13: Synchronization And Barriers

## What This Week Is

Month 4 starts with the core coordination idea: workers sometimes need to wait
for one another before they can safely continue.

## What To Read

- [../course/month-04-scans-atomics-synchronization.md](../course/month-04-scans-atomics-synchronization.md)
- [../weeks/week-12-month-03-checkpoint.md](../weeks/week-12-month-03-checkpoint.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

## Code Sketch

```python
# Sketch the smallest working version of this week's idea.
# Keep it tiny: one loop, one mask, one tile, or one benchmark.
```

Write one sentence explaining why the sketch is correct before you optimize it.

Create `results/week-13-synchronization.md` with:

- a plain-language definition of synchronization
- one example of a race condition
- one example of a barrier or “wait here” moment
- one short note on why coordination is different from math

## Write Down

Answer:

1. Why do workers need to wait sometimes?
2. What goes wrong when they do not?
3. Why is a race condition a correctness problem?

## Minimum

- the note exists
- you can explain a race condition without jargon

## Standard

- you sketch a before/after example of a barrier
- you explain why waiting can be necessary

## Stretch

- you compare synchronization to the reduction story
- you describe one debugging habit that helps with race conditions

## If You Are Behind

Keep the examples tiny. The goal is to understand the need for coordination.

## Next Week

Week 14 introduces atomics, which are one of the ways coordination can be made
explicit.
