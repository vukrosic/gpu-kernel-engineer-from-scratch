# Week 27 Triton Reductions

## Lesson Summary

Summarize how one Triton program can reduce one row or block of values.

## Key Pattern

Record the row reduction shape:

```text
row program -> column offsets -> masked load -> tl.sum or tl.max -> one output
```

## Important Detail

Explain why the masked load identity differs for sum and max.

## Limitation

Write down what changes when the row is wider than one block.

## Next Step

Write one question you want Week 28 to answer about Triton softmax.
