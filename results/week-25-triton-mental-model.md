# Week 25 Triton Mental Model

## Lesson Summary

Summarize Triton as program instances operating on blocks of data with masks.

## Key Pattern

Record the basic shape:

```text
program id -> offsets -> mask -> load block -> compute -> store block
```

## Important Detail

Explain how Triton differs from thinking one CUDA thread at a time.

## Limitation

Write down what Triton does not hide about GPU engineering.

## Next Step

Write one question you want Week 26 to answer about masks.
