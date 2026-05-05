# Week 18 Fused Row-Wise Softmax

## Lesson Summary

Summarize fused softmax as stable row-wise softmax done in one kernel-shaped
pipeline.

## Key Pattern

Record the pipeline:

```text
row max -> exponentials -> row sum -> divide -> write output
```

## Important Detail

Explain why fusion is mostly a memory-traffic story, not a change to the
softmax formula.

## Limitation

Write down what assumption the teaching kernel makes about row width.

## Next Step

Write one question you want Week 19 to answer about LayerNorm.
