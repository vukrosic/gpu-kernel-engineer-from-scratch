# Week 28: Month 07 Checkpoint

## What This Week Is

You stop and package the Triton month into a simple story. The goal is to show
that you can read Triton kernels, explain what they do, and say where the mask
or block model helped you reason clearly.

## What To Read

- [../course/month-07-triton-for-ai-kernels.md](../course/month-07-triton-for-ai-kernels.md)
- [week-27-triton-softmax.md](week-27-triton-softmax.md)

## Exact Commands

```bash
pytest tests/test_reference.py tests/test_gpu_tracks.py
python examples/reference_bench.py
```

## Build This

Write `results/month-07-checkpoint.md` as a short month summary with one block
diagram, one portfolio sentence, and one note about what is now easier to
explain than it was at the start of the month.

## Code Sketch

```python
summary = {
    "cuda_view": "threads and blocks map work to hardware",
    "triton_view": "program ids and masks map work to tiles",
}
```

Write one sentence explaining why the sketch is correct before you optimize it.

## Write Down

- What is easier to explain now?
- What still feels new or fragile?
- What should Month 8 build on?
- What is the one Triton idea you would keep in a pocket summary?

## Minimum

- one checkpoint note
- one clear summary
- one next-step question

## Standard

- compare Month 7 before and after
- list one portfolio bullet

## Stretch

- add a tiny diagram
- write one interview-style answer

## If You Are Behind

Keep the checkpoint short and practical.

## Next Week

You will move from reading Triton to using it for matmul-shaped work.
