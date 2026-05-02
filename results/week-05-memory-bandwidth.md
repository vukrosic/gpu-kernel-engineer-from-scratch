# Week 05 Memory Bandwidth

## What Was Built

Write the memory-bandwidth note that compares copy, scale, and axpy-style
work. Mention the exact input size, the benchmark helper you used, and the rough
bytes-moved story if you calculated it.

## Correctness Check

Record the reference outputs or shape checks you used to make sure the
experiments were still computing the right thing before you interpreted any
timing.

## Benchmark Or Observation

Paste the three measurements, then add one sentence about what changed between
copy, scale, and axpy. Keep the observations separate from the numbers.

## Lesson Learned

Explain, in plain language, what "memory bandwidth" means and why moving data is
often the cost that dominates the math.

## Limitation Or Next Step

Name one comparison you would like to make with larger inputs or a different
layout in Week 06.

## Write Down

- Why is copy a useful baseline?
- Why is axpy a good bridge between math and memory?
- What part of the work is arithmetic and what part is memory traffic?
- What would you expect to change if the arrays were much larger?
