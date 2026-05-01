# Week 22: Tiled Matmul

## What This Week Is

You rework matmul so each block of work is reused before you move on. This is
the first step from "it works" to "it is shaped for performance."

## What To Read

- [../course/month-06-matmul-foundations.md](../course/month-06-matmul-foundations.md)
- [week-21-naive-matmul.md](week-21-naive-matmul.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-22-tiled-matmul.md` with a sketch of your tile layout and
one note about how tiling changes reuse.

## Write Down

- What is a tile?
- What is reused inside a tile?
- What stays the same as the naive version?

## Minimum

- one tile diagram
- one note file
- one plain-language summary

## Standard

- compare tile sizes on paper
- note one tradeoff

## Stretch

- add a second tile sketch
- explain why tiling helps cache or shared memory later

## If You Are Behind

Keep one tile size and one short diagram.

## Next Week

You will connect tile size to occupancy and the cost of bigger blocks.
