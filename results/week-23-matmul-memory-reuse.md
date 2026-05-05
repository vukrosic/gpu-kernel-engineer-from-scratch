# Week 23 Matmul Memory Reuse

## Lesson Summary

Summarize why tiled matmul is mostly about reusing A and B values.

## Key Pattern

Record the path:

```text
global memory -> shared memory -> registers -> one final C write
```

## Important Detail

Explain how one loaded A or B value can contribute to multiple outputs.

## Limitation

Write down why reuse has a resource cost.

## Next Step

Write one question you want Week 24 to answer about occupancy or tile size.
