# Week 45: Benchmark Dashboard

## What This Week Is

You learn how to collect the work into a simple benchmark dashboard. The goal
is to make your progress visible without turning the repo into a spreadsheet
graveyard.

## What To Read

- [../course/month-12-portfolio-and-interviews.md](../course/month-12-portfolio-and-interviews.md)
- [week-44-month-11-checkpoint.md](week-44-month-11-checkpoint.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Create a one-page dashboard structure that records kernel name, input shape,
baseline, measured result, and the note you want future-you to remember. Leave
the values blank if you have not run them yet, but make the columns final.

## Code Sketch

```python
rows = [
    {
        "kernel": "gelu fusion",
        "shape": "batch x hidden",
        "baseline": "unfused reference",
        "result": "",
        "note": "one fewer pass over activations",
    },
    {
        "kernel": "rmsnorm",
        "shape": "batch x hidden",
        "baseline": "layernorm reference",
        "result": "",
        "note": "one reduction and one scale",
    },
]
```

The sketch is correct because it records the comparison structure first, which
is the part you need before any real numbers can mean something.

Write `results/week-45-benchmark-dashboard.md` with one dashboard sketch and
one note about what should be tracked every month.

## Write Down

- What belongs on the dashboard?
- What should be compared month to month?
- What is the simplest useful chart?

## Minimum

- one dashboard note
- one short table
- one plain-language summary

## Standard

- compare two metrics
- note one reporting rule

## Stretch

- sketch a dashboard layout
- explain one reason the dashboard matters

## If You Are Behind

Keep the dashboard to one page.

## Next Week

You will practice explaining the work as if someone asked you in an interview.
