# Week 20 RMSNorm Kernel

## Lesson Summary

Summarize RMSNorm as sum of squares, inverse RMS, weight, and output.

## Key Pattern

Record the pipeline:

```text
square -> sum -> mean square -> rsqrt -> multiply by weight
```

## Important Detail

Explain what RMSNorm removes compared with LayerNorm.

## Limitation

Write down why RMSNorm still needs a row-wise reduction.

## Next Step

Write one question you want Week 21 to answer about matmul.
