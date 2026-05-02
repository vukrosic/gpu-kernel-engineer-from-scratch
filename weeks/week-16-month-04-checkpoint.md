# Week 16: Month 04 Checkpoint

## What This Week Is

Month 4 is a checkpoint week. You consolidate synchronization, atomics, and
scan into one clean summary and make the course feel like a continuous path
again instead of three separate topics.

## What To Read

- [../course/month-04-scans-atomics-synchronization.md](../course/month-04-scans-atomics-synchronization.md)
- [../weeks/week-13-synchronization-and-barriers.md](../weeks/week-13-synchronization-and-barriers.md)
- [../weeks/week-14-atomics-and-histograms.md](../weeks/week-14-atomics-and-histograms.md)
- [../weeks/week-15-prefix-sum-and-scan.md](../weeks/week-15-prefix-sum-and-scan.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-16-month-04-checkpoint.md` as a Month 4 checkpoint note
that compares barriers, atomics, and scan in one readable story.

## Code Sketch

```python
month_4 = {
    "synchronization": "wait so shared data is ready",
    "atomics": "protect shared counters",
    "scan": "carry information forward across positions",
}
```

This sketch is correct because the checkpoint is about compressing the month
into three ideas the learner must remember, not adding a new kernel.

## Write Down

Answer:

1. What does Month 4 teach about coordination?
2. Which concept was most error-prone?
3. What would you want to remember later?
4. Which of the three topics feels most useful in GPU code?

## Minimum

- the checkpoint note exists
- the Month 4 story is summarized clearly

## Standard

- you rewrite one earlier note for clarity
- you include one short comparison between barriers, atomics, and scan

## Stretch

- you turn Month 4 into three resume bullets
- you write one paragraph about how this month prepares you for softmax

## If You Are Behind

Do not carry Month 4 confusion into Month 5. Use the checkpoint to reset.

## Next Week

Week 17 begins softmax, which turns the coordination lessons toward
transformer-adjacent math.
