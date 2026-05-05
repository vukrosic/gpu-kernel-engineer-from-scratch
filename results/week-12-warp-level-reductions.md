# Week 12 Warp-Level Reductions

## Lesson Summary

Summarize the main idea:

```text
lanes inside one warp can cooperate through shuffle operations
```

## Kernel Shape

Describe the five-part warp reduction pattern:

```text
1. one value per lane
2. exchange values with shuffle operations
3. reduce with offsets 16, 8, 4, 2, 1
4. keep the final answer in lane 0
5. write or pass onward from lane 0
```

## Important Detail

Record the difference between shared-memory block reductions and warp shuffle
reductions.

## Limitation

Write down what changes when the reduction does not fit inside one warp.

## Next Step

Write one question you want Week 13 to answer about synchronization or race
conditions.
