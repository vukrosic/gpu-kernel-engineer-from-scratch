# Week 22: Tiled Matmul

## What This Week Is

You rework matmul so each block of work is reused before you move on. This is
the first step from "it works" to "it is shaped for performance."

## What To Read

- [../course/month-06-matmul-foundations.md](../course/month-06-matmul-foundations.md)
- [../weeks/week-21-naive-matmul.md](../weeks/week-21-naive-matmul.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-22-tiled-matmul.md` with a sketch of your tile layout and
one note about how tiling changes reuse.

## Code Sketch

```python
def tiled_block(a, b, row0, col0, tile_m=4, tile_n=4, tile_k=2):
    acc = [[0.0 for _ in range(tile_n)] for _ in range(tile_m)]
    for k0 in range(0, len(a[0]), tile_k):
        for i in range(tile_m):
            for j in range(tile_n):
                for kk in range(tile_k):
                    acc[i][j] += a[row0 + i][k0 + kk] * b[k0 + kk][col0 + j]
    return acc
```

This sketch is correct because it reuses the same small submatrices before
moving on, which is the heart of tiling.

## Write Down

Answer:

1. What is a tile?
2. What is reused inside a tile?
3. What stays the same as the naive version?
4. Which dimensions are easiest to visualize?

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
