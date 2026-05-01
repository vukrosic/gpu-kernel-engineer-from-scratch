# Week 09: Naive Reductions

## What This Week Is

Week 09 starts Month 3. You move from memory movement into reductions: turning
many values into fewer values by summing or maximizing over a dimension.

## What You Need From The Repo

- [../gputriton/reference.py](../gputriton/reference.py)
- [../course/month-03-reductions.md](../course/month-03-reductions.md)
- [../weeks/week-07-timing-harness-and-benchmarking.md](../weeks/week-07-timing-harness-and-benchmarking.md)

## Exact Commands

```bash
python - <<'PY'
import numpy as np

x = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
], dtype=np.float64)

print("row sum", x.sum(axis=1))
print("row max", x.max(axis=1))
PY
```

Then run:

```bash
pytest
python examples/reference_bench.py
```

## Build This

Create `results/week-09-naive-reductions.md` and include:

- a hand-worked example of row sum
- a hand-worked example of row max
- the idea of a reduction tree or reduction chain
- why a naive reduction still teaches the right mental model

Write a tiny Python loop version of row sum in the note or in a scratch snippet,
then compare it to `numpy.sum`.

## Write Down

Answer these in the note:

1. Why is a reduction not the same as an elementwise kernel?
2. What makes row sum and row max similar?
3. Why is a reduction a coordination problem?
4. What is the output shape of a reduction over axis 1?

## Minimum

- the note exists
- you can explain row sum and row max in plain language
- you compare a loop version to NumPy

## Standard

- you draw a small reduction tree in markdown
- you explain why reductions shrink data
- you mention one correctness edge case

## Stretch

- you write a reduction helper that works for 1D and 2D arrays
- you compare loop order with reduction cost
- you explain why reductions show up everywhere in ML

## If You Are Behind

Keep the loop example tiny. The goal is to understand the shape of the problem,
not to build a polished reduction library.

## Next Week

Week 10 takes the same reduction idea and shows how a block can reduce together
using shared-memory-style thinking.
