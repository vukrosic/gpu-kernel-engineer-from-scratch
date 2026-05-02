# Week 06 Coalescing Vs Strides

## What Was Built

Write the note that compares contiguous access and strided access. Mention the
exact slicing pattern or array shape you used and why it was a fair comparison.

## Correctness Check

Record how you confirmed the two experiments were still reading the intended
elements. If the outputs differed by design, explain that difference clearly.

## Benchmark Or Observation

Write down which pattern was faster, then add one or two sentences about why
that result makes sense for cache behavior or GPU coalescing.

## Lesson Learned

Summarize the idea that memory layout is part of the algorithm. A correct kernel
can still be slow if its access pattern is unfriendly.

## Limitation Or Next Step

Name one extra pattern you would like to test, such as a transpose, and what
you would expect to learn from it.

## Write Down

- Why does contiguous access usually make life easier for hardware?
- What makes a strided pattern harder to handle?
- Why do memory layout choices matter before you think about optimization?
- What is the connection between layout and coalescing?
