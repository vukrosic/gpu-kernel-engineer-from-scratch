# Week 27: Triton Softmax

## What This Week Is

You take the softmax idea and express it as a Triton-shaped kernel. The point
is to understand where numerically stable softmax work gets fused together.

## What To Read

- [../course/month-07-triton-for-ai-kernels.md](../course/month-07-triton-for-ai-kernels.md)
- [week-26-triton-blocks-and-masks.md](week-26-triton-blocks-and-masks.md)
- [../triton_kernels/softmax.py](../triton_kernels/softmax.py)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

## Code Sketch

```python
import math

def softmax_row(xs):
    shift = max(xs)
    exps = [math.exp(x - shift) for x in xs]
    total = sum(exps)
    return [value / total for value in exps]
```

Write one sentence explaining why the sketch is correct before you optimize it.

Write `results/week-27-triton-softmax.md` with the softmax steps and one note
about stability or reuse.

Compare the Triton starter in `triton_kernels/softmax.py` against the NumPy
reference in `gputriton/reference.py`.

## Write Down

- What are the three softmax steps?
- Why is stability important?
- Where can the work be fused?

## Minimum

- one softmax step list
- one note file
- one plain-language explanation

## Standard

- compare stable and unstable versions
- note one reuse opportunity

## Stretch

- sketch a fused softmax pass
- mention one reason it matters for transformers

## If You Are Behind

Keep the three steps and one stability note.

## Next Week

You will package Month 7 and decide what is already clear enough to show.
