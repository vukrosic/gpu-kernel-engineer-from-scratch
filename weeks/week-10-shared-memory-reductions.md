# Week 10: Shared Memory Reductions

## What This Week Is

Week 10 is the first step from naive reductions toward performance-aware
reductions. You think about a block of work, partial sums, and the idea that
workers can cooperate on an intermediate result before writing one final answer.

The lesson is about structure, not syntax. You are learning to split the work
into local cooperation and a smaller final combine step.

## What You Need From The Repo

- [../course/month-03-reductions.md](../course/month-03-reductions.md)
- [../weeks/week-09-naive-reductions.md](../weeks/week-09-naive-reductions.md)
- [../gputriton/reference.py](../gputriton/reference.py)
- [../cuda/reduce_sum.cu](../cuda/reduce_sum.cu)
- [../triton_kernels/reduce_sum.py](../triton_kernels/reduce_sum.py)

## Exact Commands

```bash
python - <<'PY'
import numpy as np

def block_reduce_sum(x, block_size=4):
    partials = []
    for i in range(0, len(x), block_size):
        partials.append(x[i:i + block_size].sum())
    return np.array(partials)

x = np.arange(1, 17, dtype=np.float64)
partials = block_reduce_sum(x, block_size=4)
print("partials", partials)
print("final", partials.sum())
PY
```

Run:

```bash
pytest
python examples/reference_bench.py
```

## Build This

Create `results/week-10-shared-reductions.md` and include:

- a sketch of a block-level reduction
- the idea of partial sums and a final combine step
- a note on why shared work helps reduce repeated global memory traffic
- a comparison between naive reduction and block-style reduction

Use the `block_reduce_sum` idea above as your starting point and explain how it
would map to a GPU block using shared memory or block-local coordination.

## Code Sketch

```python
def block_reduce(values):
    while len(values) > 1:
        values = [values[i] + values[i + 1] for i in range(0, len(values), 2)]
    return values[0]
```

This sketch is correct as a reduction shape because it keeps combining pairs
until one accumulated answer remains.

## Write Down

Answer these in the note:

1. Why do partial sums help?
2. What work happens inside a block?
3. Why is shared coordination better than every worker doing everything alone?
4. How does this set you up for warp-level thinking later?

## Minimum

- the note exists
- you can explain the two-stage reduction idea
- you can describe the role of partial sums

## Standard

- you compare naive and block-style reduction
- you explain why the final combine step is smaller
- you add one sentence about memory traffic reduction

## Stretch

- you draw the reduction in three layers: input, partials, final answer
- you write a short paragraph about why this is a preview of matmul tiling
- you explain what "shared memory" is trying to avoid

## If You Are Behind

Do the block-reduction sketch and the comparison to naive reduction. That is
enough to keep the course moving.

## Next Week

Week 11 would move into warp-level thinking, but for now Month 3 is complete
enough to package and review.
