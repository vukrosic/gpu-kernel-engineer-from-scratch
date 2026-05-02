# Week 30: Autotuning

## What This Week Is

You learn why one matmul kernel is rarely the final answer. This week is about
trying more than one configuration, writing the comparison down cleanly, and
being honest about what a search space can and cannot prove.

## What To Read

- [../course/month-08-triton-matmul-and-tuning.md](../course/month-08-triton-matmul-and-tuning.md)
- [week-29-triton-matmul.md](week-29-triton-matmul.md)
- [../triton_kernels/matmul.py](../triton_kernels/matmul.py)

## Exact Commands

```bash
pytest tests/test_reference.py tests/test_gpu_tracks.py
python examples/reference_bench.py
```

## Build This

Write `results/week-30-autotuning.md` with two or three candidate settings, one
simple comparison table, and one note about which setting you would expect to
try first if you had to choose before measuring.

## Code Sketch

```python
candidate_settings = [
    {"block_m": 64, "block_n": 64, "block_k": 16},
    {"block_m": 128, "block_n": 64, "block_k": 32},
]
```

Write one sentence explaining why the sketch is correct before you optimize it.

## Write Down

- What changes when you tune?
- What costs make tuning worth it?
- How would you compare candidates fairly?
- Which parameter is the easiest one to misread?

## Minimum

- one tuning note
- one candidate table
- one plain-language summary

## Standard

- compare at least two settings
- note one reason a setting might fail

## Stretch

- propose a tiny search strategy
- explain why tuning is workload-specific

## If You Are Behind

Keep the candidate list very small.

## Next Week

You will extend the same kernel thinking to batched matmul.
