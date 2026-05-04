# Week 11: Warp-Level Thinking

## What This Week Is

You are not adding a new algorithm yet. You are learning how a group of workers
can cooperate inside a small unit of work and why that matters for reductions,
matmul, and later GPU kernels.

Think of this as moving one layer down in the coordination stack. The group is
small enough that the workers can act together, but the result is still part of
a larger reduction.

## What To Read

- [../course/month-03-reductions.md](../course/month-03-reductions.md)
- [../weeks/week-10-naive-reduction-kernels.md](../weeks/week-10-naive-reduction-kernels.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

Try a tiny grouped reduction in Python:

```bash
python - <<'PY'
def group_reduce(values):
    partials = []
    for i in range(0, len(values), 2):
        partials.append(values[i] + values[i + 1])
    return partials

values = [1, 2, 3, 4, 5, 6, 7, 8]
print(group_reduce(values))
PY
```

## Build This

Create `results/week-11-warp-thinking.md` with:

- a sketch of a warp or small worker group
- a short explanation of why partial work matters
- one example of a reduction tree
- one note on how this previews better reduction code later

## Code Sketch

```python
def group_reduce(values):
    partials = []
    for i in range(0, len(values), 2):
        partials.append(values[i] + values[i + 1])
    return partials
```

This sketch is correct because the group first reduces nearby values into
partials instead of forcing one worker to do every step alone.

## Write Down

Answer:

1. What is a warp trying to coordinate?
2. Why do grouped workers matter for reductions?
3. How is this different from a single-thread loop?

## Minimum

- the note exists
- you can explain warp-level thinking in your own words

## Standard

- you draw a small worker-group sketch
- you connect it back to Week 10

## Stretch

- you explain where warp thinking would help in matmul
- you compare a loop reduction to a grouped reduction

## If You Are Behind

Keep the note short and focus on the idea, not the hardware details.

## Next Week

Week 12 is the Month 3 checkpoint, where you package the reduction story.
