# Week 23: Tiling And Occupancy

## What This Week Is

You connect tile size to how much work fits on the GPU at once. The point is
to understand why bigger is not always better.

## What To Read

- [../course/month-06-matmul-foundations.md](../course/month-06-matmul-foundations.md)
- [week-22-tiled-matmul.md](week-22-tiled-matmul.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-23-tiling-and-occupancy.md` with one table of tile sizes
and one note about reuse versus parallelism.

## Write Down

- What does occupancy mean in your own words?
- Why can a huge tile hurt performance?
- What would you measure first?

## Minimum

- one occupancy note
- one table or sketch
- one short explanation

## Standard

- compare two tile sizes
- note one bottleneck

## Stretch

- connect tile size to register or shared-memory pressure
- propose a better tile choice

## If You Are Behind

Keep the comparison to two tile sizes.

## Next Week

You will pause for a month checkpoint and package the matmul work.
