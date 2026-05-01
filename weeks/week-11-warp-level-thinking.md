# Week 11: Warp-Level Thinking

## What This Week Is

You are not adding a new algorithm yet. You are learning how a group of workers
can cooperate inside a small unit of work and why that matters for reductions,
matmul, and later GPU kernels.

## What To Read

- [../course/month-03-reductions.md](../course/month-03-reductions.md)
- [../weeks/week-10-shared-memory-reductions.md](../weeks/week-10-shared-memory-reductions.md)

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

Create `results/week-11-warp-thinking.md` with:

- a sketch of a warp or small worker group
- a short explanation of why partial work matters
- one example of a reduction tree
- one note on how this previews better reduction code later

## Write Down

Answer:

1. What is a warp trying to coordinate?
2. Why do grouped workers matter for reductions?
3. How is this different from a single-thread loop?

## Minimum

- the note exists
- you can explain warp-level thinking in your own words

## Standard

- you draw a small worker-group sketch
- you connect it back to Week 10

## Stretch

- you explain where warp thinking would help in matmul
- you compare a loop reduction to a grouped reduction

## If You Are Behind

Keep the note short and focus on the idea, not the hardware details.

## Next Week

Week 12 is the Month 3 checkpoint, where you package the reduction story.
