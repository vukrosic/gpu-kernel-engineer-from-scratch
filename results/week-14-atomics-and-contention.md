# Week 14 Atomics And Contention

## Lesson Summary

Summarize why atomics are needed when many threads update the same memory
location.

## Key Pattern

Record the histogram update shape:

```text
read input value
choose bin
atomicAdd counter
```

## Important Detail

Explain the difference between a barrier and an atomic operation.

## Limitation

Write down how contention can make correct atomic code slow.

## Next Step

Write one question you want Week 15 to answer about scan or offsets.
