# Week 23: Tiling And Occupancy

## What This Week Is

You connect tile size to how much work fits on the GPU at once. The point is
to understand why bigger is not always better and why the resource story matters
as much as reuse.

## What To Read

- [../course/month-06-matmul-foundations.md](../course/month-06-matmul-foundations.md)
- [../weeks/week-22-tiled-matmul.md](../weeks/week-22-tiled-matmul.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-23-tiling-occupancy.md` with one table of tile sizes and
one note about reuse versus parallelism.

## Code Sketch

```python
def tile_report(candidates):
    rows = []
    for tile_m, tile_n, tile_k in candidates:
        rows.append({
            "tile": f"{tile_m}x{tile_n}x{tile_k}",
            "reuse": tile_k,
            "pressure": tile_m * tile_n,
        })
    return rows
```

This sketch is correct because it turns tile choice into an explicit comparison
instead of a guess, which is what you want before tuning.

## Write Down

Answer:

1. What does occupancy mean in your own words?
2. Why can a huge tile hurt performance?
3. What would you measure first?
4. Which tile dimension feels most expensive?

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
