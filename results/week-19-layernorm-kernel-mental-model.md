# Week 19 LayerNorm Kernel Mental Model

## Lesson Summary

Summarize LayerNorm as row-wise mean, variance, normalization, scale, and bias.

## Key Pattern

Record the pipeline:

```text
row sum -> mean -> squared differences -> variance -> normalize -> gamma/beta
```

## Important Detail

Explain why LayerNorm is both reduction-like and elementwise.

## Limitation

Write down why keeping row values close can reduce memory traffic but increase
resource pressure.

## Next Step

Write one question you want Week 20 to answer about RMSNorm.
