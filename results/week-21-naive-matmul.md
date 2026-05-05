# Week 21 Naive Matrix Multiplication

## Lesson Summary

Summarize naive matmul as one output element per thread and one dot product per
output.

## Key Pattern

Record the index formulas:

```text
A[row * K + p]
B[p * N + col]
C[row * N + col]
```

## Important Detail

Explain why rectangular shapes are important for correctness tests.

## Limitation

Write down what values the naive kernel reloads too often.

## Next Step

Write one question you want Week 22 to answer about tiling.
