# Week 47: Resume And Story

## What This Week Is

You turn the year of work into a clear project story for a resume and GitHub
profile. The goal is to make the output easy for another engineer to scan and
trust.

## What To Read

- [../course/month-12-portfolio-and-interviews.md](../course/month-12-portfolio-and-interviews.md)
- [week-46-interview-explanations.md](week-46-interview-explanations.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Draft three resume bullets and one short project summary paragraph. Make the
bullets specific enough to show the kernels, the validation work, and the shape
of the performance story.

## Code Sketch

```python
story = {
    "problem": "make transformer-adjacent kernels easier to understand and measure",
    "approach": "build reference-first implementations, then simplify or fuse them",
    "proof": "keep benchmarks, notes, and correctness checks together",
}
```

The sketch is correct because it captures the same three claims a resume reader
needs: what you built, how you proved it, and why it matters.

Write `results/week-47-resume-and-story.md` with three resume bullets and one
short project summary paragraph.

## Write Down

- What is the headline for the project?
- What three bullets are strongest?
- What should be linked from the README?

## Minimum

- one project summary
- three resume bullets
- one plain-language explanation

## Standard

- compare two summary versions
- note one thing to cut

## Stretch

- draft a GitHub profile blurb
- write one LinkedIn-style paragraph

## If You Are Behind

Keep the bullets short and specific.

## Next Week

You will finish the capstone and turn the repo into a finished public artifact.
