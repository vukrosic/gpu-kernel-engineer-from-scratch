# Week 36: Month 09 Checkpoint

## What This Week Is

You stop and package the integration month. The point is to decide what you can
now explain about baselines, wrappers, and testing, and to turn that into a
clean story a reviewer could follow.

## What To Read

- [../course/month-09-pytorch-integration.md](../course/month-09-pytorch-integration.md)
- [week-35-gpu-test-matrix.md](week-35-gpu-test-matrix.md)

## Exact Commands

```bash
pytest tests/test_reference.py tests/test_gpu_tracks.py
python examples/reference_bench.py
```

## Build This

Write `results/month-09-checkpoint.md` with the three clearest lessons from
Month 9 and one sentence about how custom kernels fit into a PyTorch workflow.

## Code Sketch

```python
summary = {
    "baseline": "reference first, then compare",
    "wrapper": "small boundary, visible kernel",
    "tests": "shape, dtype, and edge cases all matter",
}
```

Write one sentence explaining why the sketch is correct before you optimize it.

## Write Down

- What is easier to explain now?
- What still needs more practice?
- What should Month 10 focus on?
- What is the one integration lesson you would keep in a summary note?

## Minimum

- one checkpoint note
- one portfolio summary
- one next-step question

## Standard

- compare Month 9 before and after
- add one resume bullet idea

## Stretch

- add a tiny table or diagram
- write one interview-ready answer

## If You Are Behind

Keep the checkpoint short and practical.

## Next Week

You will move to transformer kernels and the first fused activation work.
