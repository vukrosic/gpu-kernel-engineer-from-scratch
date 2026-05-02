# Week 14: Atomics And Histograms

## What This Week Is

You are now on the "one at a time" side of coordination. Atomics make the
shared update explicit, and histograms are a good way to see why that matters:
many inputs want to touch a small set of counters, so contention shows up fast.

## What To Read

- [../course/month-04-scans-atomics-synchronization.md](../course/month-04-scans-atomics-synchronization.md)
- [../weeks/week-13-synchronization-and-barriers.md](../weeks/week-13-synchronization-and-barriers.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-14-atomics-histograms.md` with a tiny histogram example, a
plain description of what the atomic protects, and one note about the tradeoff
between simplicity and contention.

## Code Sketch

```python
def histogram(values, bins=4):
    counts = [0] * bins
    for value in values:
        bin_idx = value % bins
        counts[bin_idx] += 1
    return counts
```

This sketch is correct because every input maps to exactly one bucket, which
makes the shared-counter problem visible before you add atomics.

## Write Down

Answer:

1. Why do many workers updating one counter create trouble?
2. Why are histograms a natural atomic example?
3. When is an atomic approach useful even if it is not the fastest?
4. What changes if the input is badly skewed toward one bin?

## Minimum

- the note exists
- you can explain atomics in one paragraph

## Standard

- you compare atomics to barriers
- you include one small histogram example and its bucket layout

## Stretch

- you explain where atomics are likely to hurt performance
- you compare a histogram to a reduction or a scan

## If You Are Behind

Use a small histogram with three or four buckets and keep the reasoning simple.

## Next Week

Week 15 moves to prefix sums, where the shape of the data changes again.
