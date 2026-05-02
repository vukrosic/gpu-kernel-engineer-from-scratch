# Week 25: Triton Mental Model

## What This Week Is

You learn how Triton thinks about programs, blocks, and masks. The point is not
to memorize syntax yet; it is to be able to look at a kernel and describe what
one program instance owns, how it indexes data, and why the edge cases stay
safe.

## What To Read

- [../course/month-07-triton-for-ai-kernels.md](../course/month-07-triton-for-ai-kernels.md)
- [../triton/README.md](../triton/README.md)
- [../triton_kernels/vector_add.py](../triton_kernels/vector_add.py)
- [../gputriton/reference.py](../gputriton/reference.py)
- [week-24-month-06-checkpoint.md](week-24-month-06-checkpoint.md)

## Exact Commands

```bash
pytest tests/test_reference.py tests/test_gpu_tracks.py
python examples/reference_bench.py
```

## Build This

Write `results/week-25-triton-mental-model.md` as a Triton vocabulary note
with one program/block/mask diagram, one vector-add trace, and one CUDA-to-Triton
comparison in plain language.

## Code Sketch

```python
def program_for_block(pid, block_size, n):
    start = pid * block_size
    offsets = [start + i for i in range(block_size)]
    mask = [offset < n for offset in offsets]
    return offsets, mask
```

Write one sentence explaining why the sketch is correct before you optimize it.

## Write Down

- What is a program instance responsible for?
- What does a block own?
- Why does a mask exist at the boundary?
- How would you explain this to someone who knows CUDA but not Triton?

## Minimum

- one vocabulary note
- one block diagram
- one plain-language summary

## Standard

- compare Triton blocks to CUDA blocks
- explain one mask example from the vector-add shape

## Stretch

- sketch a tiny elementwise kernel in pseudocode
- mention one reason Triton is readable once the model clicks

## If You Are Behind

Keep the vocabulary note and the diagram, then move on.

## Next Week

You will use the same mental model to describe edge blocks and safe masking.
