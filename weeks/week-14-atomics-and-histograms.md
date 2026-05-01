# Week 14: Atomics And Histograms

## What This Week Is

You are now looking at the “one at a time” side of coordination: atomics.
Histograms make the idea concrete because lots of inputs want to update a small
number of counters.

## What To Read

- [../course/month-04-scans-atomics-synchronization.md](../course/month-04-scans-atomics-synchronization.md)
- [../weeks/week-13-synchronization-and-barriers.md](../weeks/week-13-synchronization-and-barriers.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Create `results/week-14-atomics-histograms.md` and include:

- what an atomic update is trying to protect
- why a histogram is a good example
- what the tradeoff is between simplicity and speed

## Write Down

Answer:

1. Why do many workers updating one counter create trouble?
2. Why are histograms a natural atomic example?
3. When is an atomic approach useful even if it is not the fastest?

## Minimum

- the note exists
- you can explain atomics in one paragraph

## Standard

- you compare atomics to barriers
- you include one small histogram example

## Stretch

- you explain where atomics are likely to hurt performance
- you compare a histogram to a reduction

## If You Are Behind

Use a small histogram with three or four buckets and keep the reasoning simple.

## Next Week

Week 15 moves to prefix sums, where the shape of the data changes again.
