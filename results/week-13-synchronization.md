# Week 13 Synchronization And Barriers

## Lesson Summary

Summarize why threads need barriers when one thread reads data written by
another thread in the same block.

## Key Pattern

Record the pattern:

```text
write shared memory
__syncthreads()
read shared memory
```

## Important Detail

Explain why every thread in the block must reach the same barrier.

## Limitation

Write down why `__syncthreads()` does not synchronize different blocks.

## Next Step

Write one question you want Week 14 to answer about shared updates.
