# Week 08: Month 02 Checkpoint

## What This Week Is

Week 08 is a packaging week. You are not learning a new memory concept. You are
wrapping Month 2 into a clean summary so the next month can start fresh.

The checkpoint should show that you know the difference between a noisy number
and a useful measurement, and that you can explain why memory movement kept
showing up in different disguises.

## What You Need From The Repo

- [../course/month-02-memory-and-benchmarking.md](../course/month-02-memory-and-benchmarking.md)
- [../weeks/week-05-memory-bandwidth-and-axpy.md](../weeks/week-05-memory-bandwidth-and-axpy.md)
- [../weeks/week-06-coalescing-vs-strides.md](../weeks/week-06-coalescing-vs-strides.md)
- [../weeks/week-07-timing-harness-and-benchmarking.md](../weeks/week-07-timing-harness-and-benchmarking.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

Then open the notes you wrote in Weeks 05-07 and clean them up.

## Build This

Create `results/week-08-month-02-checkpoint.md` with these sections:

- the biggest memory lesson you learned
- one experiment that showed a clear difference
- one thing you now trust more about benchmarking
- one thing you still want to improve
- one sentence about how Month 3 will change the problem

## Code Sketch

```python
month_02 = {
    "lesson": "memory movement often dominates the work",
    "comparison": "contiguous versus strided access",
    "habit": "repeat timings and trust the median",
    "next": "reductions",
}
```

This sketch is correct because the checkpoint is a summary artifact, not a new
algorithm. It exists to turn observations into a story you can reuse.

## Write Down

Answer these in the checkpoint note:

1. What is the most important memory idea so far?
2. What was the noisiest benchmark?
3. What did you learn about comparing results?
4. What would you tell someone starting Month 2 tomorrow?

## Minimum

- `results/week-08-month-02-checkpoint.md` exists
- Weeks 05-07 notes are readable
- you can explain Month 2 in five sentences

## Standard

- you include one before/after comparison
- you include one chart or table in markdown
- you write one paragraph on what benchmark discipline means

## Stretch

- you turn the Month 2 checkpoint into three resume bullets
- you add one section called "what I would do differently"
- you write a short bridge paragraph from memory to reductions

## If You Are Behind

Do not start Month 3 until you can explain why memory movement matters and why
your benchmark numbers deserve trust.

## Next Week

Week 09 begins reductions, where many values get combined into fewer values.
