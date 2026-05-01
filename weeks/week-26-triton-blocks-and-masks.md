# Week 26: Triton Blocks And Masks

## What This Week Is

You practice thinking in Triton blocks and masks on small examples. The goal
is to see how one kernel can cover a whole row or vector safely.

## What To Read

- [../course/month-07-triton-for-ai-kernels.md](../course/month-07-triton-for-ai-kernels.md)
- [week-25-triton-mental-model.md](week-25-triton-mental-model.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-26-triton-blocks-and-masks.md` with one worked example of
an edge block and its mask.

## Write Down

- What happens when a block is partly outside the data?
- How does a mask protect correctness?
- What pattern repeats across rows?

## Minimum

- one edge-case example
- one note file
- one sentence on masks

## Standard

- compare two shapes or row lengths
- note one failure mode without masks

## Stretch

- sketch a masked elementwise kernel
- explain why the code stays simple

## If You Are Behind

Keep just one example and one diagram.

## Next Week

You will use the same Triton model to describe a softmax kernel.
