# Week 26: Triton Blocks And Masks

## What This Week Is

You practice thinking in Triton blocks and masks on small examples. The goal is
to see how one kernel can safely cover a whole row or vector, even when the
last block is only partially full.

## What To Read

- [../course/month-07-triton-for-ai-kernels.md](../course/month-07-triton-for-ai-kernels.md)
- [week-25-triton-mental-model.md](week-25-triton-mental-model.md)
- [../triton_kernels/vector_add.py](../triton_kernels/vector_add.py)
- [../triton_kernels/softmax.py](../triton_kernels/softmax.py)

## Exact Commands

```bash
pytest tests/test_reference.py tests/test_gpu_tracks.py
python examples/reference_bench.py
```

## Build This

Write `results/week-26-triton-blocks-masks.md` with one worked edge-block
example, one mask table, and one sentence that shows why masked results still
match the reference.

## Code Sketch

```python
def masked_load(values, start, block_size):
    block = []
    for i in range(block_size):
        idx = start + i
        block.append(values[idx] if idx < len(values) else 0.0)
    return block
```

Write one sentence explaining why the sketch is correct before you optimize it.

## Write Down

- What happens when a block runs past the end of the row?
- How does the mask protect correctness?
- What pattern repeats across rows or tiles?
- What would break if the mask were omitted?

## Minimum

- one edge-case example
- one note file
- one sentence on masks

## Standard

- compare two shapes or row lengths
- note one failure mode without masks

## Stretch

- sketch a masked elementwise kernel
- explain why the code stays simple even when the row length is uneven

## If You Are Behind

Keep one worked example and one diagram.

## Next Week

You will use the same Triton model to describe a softmax kernel.
