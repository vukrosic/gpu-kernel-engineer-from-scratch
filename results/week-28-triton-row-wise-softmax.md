# Week 28 Triton Row-Wise Softmax

## Lesson Summary

Summarize Triton softmax as one row program with max, exp, sum, divide, and a
masked store.

## Key Pattern

Record the pipeline:

```text
masked load -> tl.max -> exp -> tl.sum -> divide -> masked store
```

## Important Detail

Explain why invalid max positions use negative infinity.

## Limitation

Write down what changes when a row does not fit in one block.

## Next Step

Write one question you want Week 29 to answer about Triton matmul.
