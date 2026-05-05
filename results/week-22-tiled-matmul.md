# Week 22 Tiled Matrix Multiplication

## Lesson Summary

Summarize tiling as loading small chunks of A and B so a block can reuse them
while computing a C tile.

## Key Pattern

Record the loop shape:

```text
for each K tile:
  load A tile
  load B tile
  wait
  accumulate
  wait
```

## Important Detail

Explain why the two barriers around tile use are needed.

## Limitation

Write down why real tiled kernels need boundary checks.

## Next Step

Write one question you want Week 23 to answer about tile size and occupancy.
