# Week 15: Prefix Sum And Scan

## What This Week Is

Scan is one of the classic building blocks in parallel programming. It looks
like a simple cumulative sum, but it teaches you how information can flow
across many positions instead of only collapsing into one result. That makes
it a useful bridge between reductions and more structured GPU algorithms.

## What To Read

- [../course/month-04-scans-atomics-synchronization.md](../course/month-04-scans-atomics-synchronization.md)
- [../weeks/week-14-atomics-and-histograms.md](../weeks/week-14-atomics-and-histograms.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-15-scan.md` with a hand-worked scan example, a clear note
about inclusive versus exclusive scan, and one sentence about why scan is a
useful GPU pattern.

## Code Sketch

```python
def prefix_sum(values):
    out = []
    running = 0
    for value in values:
        running += value
        out.append(running)
    return out
```

This sketch is correct because it records the running total at every step, so
the output keeps the same length as the input.

## Write Down

Answer:

1. What does a scan produce?
2. Why is it different from a reduction?
3. Why do parallel systems care about scan?
4. What is one place where inclusive and exclusive scan differ?

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
