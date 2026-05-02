# Week 04: Month 01 Checkpoint

## What This Week Is

Week 04 is not a new concept week. It is a checkpoint. You review the first
three weeks, clean up your notes, and package the Month 1 story so far.

The point is to make the month readable from the outside. If someone opened your
notes tomorrow, they should be able to see the arc from mental model to vector
add to indexing without needing a live explanation from you.

## What You Need From The Repo

- [../course/month-01-gpu-foundations.md](../course/month-01-gpu-foundations.md)
- [../weeks/week-01-gpu-mental-model.md](../weeks/week-01-gpu-mental-model.md)
- [../weeks/week-02-gpu-setup-and-vector-add.md](../weeks/week-02-gpu-setup-and-vector-add.md)
- [../weeks/week-03-grids-blocks-threads-and-indexing.md](../weeks/week-03-grids-blocks-threads-and-indexing.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

Then open the notes you already wrote in `results/` and clean them up.

## Build This

Create `results/week-04-month-01-checkpoint.md` with these sections:

- what you understand about CPU vs GPU now
- what vector add taught you
- what indexing taught you
- what still feels fuzzy
- what you want Month 2 to answer

Also rewrite your Week 01 and Week 02 notes so they are easier to read.

## Code Sketch

```python
month_01 = {
    "mental_model": "CPU and GPU solve different shapes of work well",
    "first_kernel": "vector add proved the one-worker-one-output idea",
    "indexing": "layout and worker mapping must agree",
}
```

This sketch is correct because a checkpoint is a summary artifact, not a new
algorithm. It captures the month in compact form before you move on.

## Write Down

Answer these in the checkpoint note:

1. What is the clearest GPU idea you learned in Month 1?
2. What was the hardest part so far?
3. What would you explain differently if you taught this again?
4. What benchmark result do you trust most?

## Minimum

- `results/week-04-month-01-checkpoint.md` exists
- your Week 01 and Week 02 notes are readable
- you can explain Month 1 in five sentences

## Standard

- you summarize the Month 1 roadmap in a short paragraph
- you include one benchmark observation
- you include one question for Month 2

## Stretch

- you turn Month 1 into three resume bullets
- you add a small before/after note about confidence in indexing
- you write one short paragraph explaining why the checkpoint week exists

## If You Are Behind

Do not start Month 2 until Month 1 is written down cleanly. The checkpoint week
is there to stop the course from becoming a pile of half-finished ideas.

## Next Week

Week 05 starts Month 2 and shifts the focus from indexing to memory movement.
