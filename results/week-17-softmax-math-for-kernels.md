# Week 17 Softmax Math For Kernels

## Lesson Summary

Summarize softmax as row max, exponentials, row sum, and normalization.

## Key Pattern

Record the stable softmax steps:

```text
max
subtract max
exp
sum
divide
```

## Important Detail

Explain why subtracting the row max prevents overflow without changing the
final probabilities.

## Limitation

Write down why softmax is simple as a formula but more complex as a GPU kernel.

## Next Step

Write one question you want Week 18 to answer about fused softmax.
