# Week 11 Block-Level Reductions With Shared Memory

## Lesson Summary

Summarize the main idea of the week:

```text
one block cooperates to reduce many input values into one partial or final output
```

## Kernel Shape

Describe the six-part block reduction pattern:

```text
1. map one block to one output region
2. load values
3. store them in shared memory
4. synchronize
5. reduce in stages
6. write one result
```

## Important Detail

Record why `__syncthreads()` is needed after loading shared memory and after
each reduction stage.

## Limitation

Write down when this simple version is not enough, especially when a row is
larger than one block.

## Next Step

Write one question you want Week 12 to answer about warp-level reductions.
