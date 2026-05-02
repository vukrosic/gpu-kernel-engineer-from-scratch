# Week 46: Interview Explanations

## What This Week Is

You turn the course into words you can say out loud. The goal is to explain the
work clearly, not to memorize canned answers.

## What To Read

- [../course/month-12-portfolio-and-interviews.md](../course/month-12-portfolio-and-interviews.md)
- [week-45-benchmark-dashboard.md](week-45-benchmark-dashboard.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write three interview-style answers: one about speeding up a kernel, one about a
memory tradeoff, and one about a bug or mismatch you solved. Keep them short
enough to practice without notes.

## Code Sketch

```python
answers = {
    "how do you speed up attention?": [
        "reduce memory traffic",
        "reuse tiles or cached data",
        "verify against a reference path",
    ],
    "what tradeoff did KV cache introduce?": [
        "lower decode cost",
        "higher memory use",
        "more state to manage",
    ],
}
```

The sketch is correct because it turns the week into a question-and-answer bank
you can rehearse and refine.

Write `results/week-46-interview-explanations.md` with three interview-style
answers and one note about what you still want to practice.

## Write Down

- How do you explain your strongest kernel?
- How do you explain one performance tradeoff?
- How do you explain one bug you solved?

## Minimum

- one interview note
- one answer draft
- one short summary

## Standard

- compare two answers
- note one weak spot

## Stretch

- practice a verbal summary
- explain one project in 60 seconds

## If You Are Behind

Keep the answers short and honest.

## Next Week

You will shape the project story and resume bullets that tie the year together.
