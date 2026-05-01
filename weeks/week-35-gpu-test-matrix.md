# Week 35: GPU Test Matrix

## What This Week Is

You think through the combinations of shape, dtype, and device that matter for
testing. The goal is to stop relying on one lucky example.

## What To Read

- [../course/month-09-pytorch-integration.md](../course/month-09-pytorch-integration.md)
- [week-34-custom-op-wrapper.md](week-34-custom-op-wrapper.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

## Code Sketch

```python
# Sketch the smallest working version of this week's idea.
# Keep it tiny: one loop, one mask, one tile, or one benchmark.
```

Write one sentence explaining why the sketch is correct before you optimize it.

Write `results/week-35-gpu-test-matrix.md` with a small test matrix and one
note about which cases are most important.

## Write Down

- Which cases are mandatory?
- Which cases are optional?
- What bug would the matrix catch first?

## Minimum

- one matrix note
- one short table
- one explanation in plain language

## Standard

- compare three cases
- note one missing edge case

## Stretch

- add a dtype or shape expansion
- explain one failure mode

## If You Are Behind

Keep the matrix to the most important cases only.

## Next Week

You will package Month 9 and turn the integration work into a clean checkpoint.
