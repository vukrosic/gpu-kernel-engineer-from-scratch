# Week 12: Month 03 Checkpoint

## What This Week Is

This is the Month 3 checkpoint. You clean up your reduction notes and turn the
first three months into a readable story.

The checkpoint should make the month feel complete: naive reductions, block
thinking, and warp-level thinking should all be present as one connected arc.

## What To Read

- [../course/month-03-reductions.md](../course/month-03-reductions.md)
- [../weeks/week-09-naive-reductions.md](../weeks/week-09-naive-reductions.md)
- [../weeks/week-10-shared-memory-reductions.md](../weeks/week-10-shared-memory-reductions.md)
- [../weeks/week-11-warp-level-thinking.md](../weeks/week-11-warp-level-thinking.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Create `results/week-12-month-03-checkpoint.md` with:

- what reductions are
- why grouped work matters
- what got clearer over the last four weeks
- what still feels fuzzy
- one sentence that previews Month 4

## Code Sketch

```python
summary = {
    "reduction": "turn many values into fewer values",
    "grouped_work": "workers cooperate on partial sums",
    "next": "synchronization",
}
```

This sketch is correct because a checkpoint is a summary artifact, not a new
algorithm. It captures the lesson in compact form before you move on.

## Write Down

Answer:

1. What is the main lesson from Month 3?
2. Which reduction idea felt hardest?
3. What would you tell someone starting now?

## Minimum

- the checkpoint note exists
- Month 3 is summarized in your own words

## Standard

- you rewrite one earlier note so it is easier to read
- you include one benchmark or experiment observation

## Stretch

- you turn Month 3 into a short resume bullet set
- you write a one-paragraph bridge to synchronization

## If You Are Behind

Do not start Month 4 until Month 3 is cleaned up. The checkpoint is the point.

## Next Week

Week 13 begins synchronization, where coordination between threads becomes the
main problem.
