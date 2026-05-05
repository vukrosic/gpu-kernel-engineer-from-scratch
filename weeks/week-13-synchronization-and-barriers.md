# Week 13: Synchronization And Barriers

## What This Week Is

Month 4 starts with the coordination problem: multiple workers can be right on
their own and still be wrong together if one of them reads shared state too
early. This week is about naming that risk, showing the "wait here" moment, and
building the habit of thinking about correctness before throughput.

## What To Read

- [../course/month-04-scans-atomics-synchronization.md](../course/month-04-scans-atomics-synchronization.md)
- [../weeks/week-12-warp-level-reductions.md](../weeks/week-12-warp-level-reductions.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-13-synchronization.md` with a tiny barrier example, a race
condition explained in plain language, and one note about why waiting is a
correctness tool rather than a math trick.

## Code Sketch

```python
def wait_for_writer(shared):
    if shared["id"] == 0:
        shared["value"] = shared["left"] + shared["right"]
        shared["ready"] = True

    while not shared["ready"]:
        pass

    return shared["value"]
```

This sketch is correct because it shows one worker publishing a value and the
other workers refusing to read it until the publish flag is set.

## Write Down

Answer:

1. What has to be true before another worker can continue?
2. What breaks when a worker reads too early?
3. Why is a race condition a correctness problem and not just a slowdown?
4. What is the smallest barrier example you can explain out loud?

## Minimum

- the note exists
- you can explain a race condition without jargon

## Standard

- you sketch a before/after barrier example
- you explain why waiting can be necessary even when the math is simple

## Stretch

- you compare synchronization to the reduction story
- you describe one debugging habit that helps you spot races faster

## If You Are Behind

Keep the example tiny. The goal is to understand why coordination matters.

## Next Week

Week 14 introduces atomics, which make the "one at a time" side of
coordination explicit.
