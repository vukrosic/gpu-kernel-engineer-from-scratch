# Week 20: Month 05 Checkpoint

## What This Week Is

Month 5 is a checkpoint week. You turn softmax, fusion, and LayerNorm into a
clean summary and get ready to pivot into matmul.

## What To Read

- [../course/month-05-softmax-and-normalization.md](../course/month-05-softmax-and-normalization.md)
- [../weeks/week-17-softmax-math-for-kernels.md](../weeks/week-17-softmax-math-for-kernels.md)
- [../weeks/week-18-fused-softmax.md](../weeks/week-18-fused-softmax.md)
- [../weeks/week-19-layernorm.md](../weeks/week-19-layernorm.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-20-month-05-checkpoint.md` as a Month 5 checkpoint note
that compares softmax, fusion, and LayerNorm in one readable summary.

## Code Sketch

```python
month_5 = {
    "softmax": "stable scores into probabilities",
    "fusion": "fewer passes over the same data",
    "layernorm": "normalize with shared statistics",
}
```

This sketch is correct because a checkpoint should compress the month into the
three ideas the learner must remember.

## Write Down

Answer:

1. What is the main lesson from Month 5?
2. Which normalization concept felt most important?
3. How do softmax and LayerNorm fit the roadmap?
4. What changed in your thinking about memory traffic?

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
