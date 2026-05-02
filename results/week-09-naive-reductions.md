# Week 09 Naive Reductions

## What Was Built

Write the note that explains row sum and row max, including at least one
hand-worked example. Mention the row you reduced and the output shape you got.

## Correctness Check

Record the loop version and the NumPy version you compared. Add a sentence
about why you trusted the reference output.

## Benchmark Or Observation

Describe any timing or shape observation you saw when you compared a loop to
NumPy. If you did not time anything, note the reduction tree or chain shape you
drew instead.

## Lesson Learned

Summarize why a reduction is different from an elementwise kernel and why the
output gets smaller as the work proceeds.

## Limitation Or Next Step

Write one thing that naive reductions still do not solve and what Week 10 is
trying to improve.

## Write Down

- Why is a reduction not the same as an elementwise kernel?
- What makes row sum and row max similar?
- Why is a reduction a coordination problem?
- What is the output shape of a reduction over axis 1?
