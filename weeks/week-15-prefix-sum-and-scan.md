# Week 15: Prefix Sum And Scan

## What This Week Is

Scan is one of the classic building blocks in parallel programming. It looks
like a simple cumulative sum, but it teaches you how information can flow across
many positions instead of only collapsing into one result.

## What To Read

- [../course/month-04-scans-atomics-synchronization.md](../course/month-04-scans-atomics-synchronization.md)
- [../weeks/week-14-atomics-and-histograms.md](../weeks/week-14-atomics-and-histograms.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Create `results/week-15-prefix-sum.md` with:

- a hand-worked prefix sum example
- the difference between inclusive and exclusive scan
- one note on why scan is useful in GPU algorithms

## Write Down

Answer:

1. What does a scan produce?
2. Why is it different from a reduction?
3. Why do parallel systems care about scan?

## Minimum

- the note exists
- you can explain inclusive scan in plain language

## Standard

- you compare inclusive and exclusive scan
- you give one example of where scan could be useful

## Stretch

- you sketch how a scan could be done in stages
- you connect scan back to reductions and histograms

## If You Are Behind

Keep the worked example small. The point is the data flow, not the size.

## Next Week

Week 16 is the Month 4 checkpoint, where you package synchronization, atomics,
and scan into one readable progress note.
