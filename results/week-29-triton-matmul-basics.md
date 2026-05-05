# Week 29 Triton Matmul Basics

## Lesson Summary

Summarize one Triton matmul program as one C tile with a K loop.

## Key Pattern

Record the tile ownership:

```text
program ids -> C tile -> A tile + B tile -> tl.dot -> store C tile
```

## Important Detail

Explain what BLOCK_M, BLOCK_N, and BLOCK_K control.

## Limitation

Write down why real kernels need masks for M, N, and K.

## Next Step

Write one question you want Week 30 to answer about matmul performance knobs.
