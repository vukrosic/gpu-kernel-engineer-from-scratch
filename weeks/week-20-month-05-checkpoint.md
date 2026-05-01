# Week 20: Month 05 Checkpoint

## What This Week Is

Month 5 is a checkpoint week. You turn softmax, fusion, and LayerNorm into a
clean summary and get ready to pivot into matmul.

## What To Read

- [../course/month-05-softmax-and-normalization.md](../course/month-05-softmax-and-normalization.md)
- [../weeks/week-17-softmax-math.md](../weeks/week-17-softmax-math.md)
- [../weeks/week-18-fused-softmax.md](../weeks/week-18-fused-softmax.md)
- [../weeks/week-19-layernorm.md](../weeks/week-19-layernorm.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Create `results/month-05-checkpoint.md` with:

- the biggest normalization lesson you learned
- one thing that became clearer about fusion
- one sentence about why these ideas matter for transformers
- one bridge sentence to matmul

## Write Down

Answer:

1. What is the main lesson from Month 5?
2. Which normalization concept felt most important?
3. How do softmax and LayerNorm fit the roadmap?

## Minimum

- the checkpoint note exists
- Month 5 is summarized clearly

## Standard

- you rewrite one of your earlier notes
- you include one paragraph on memory traffic or fusion

## Stretch

- you turn Month 5 into a short interview answer
- you write one line on how this feeds into attention

## If You Are Behind

Do not carry Month 5 confusion into matmul. Use the checkpoint to reset.

## Next Week

Week 21 begins matmul, the biggest compute primitive in the roadmap so far.
