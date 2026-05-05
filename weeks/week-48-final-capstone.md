# Week 48: Final Capstone

Week 48 finishes the course.

The final capstone is the public shape of the project:

```text
what the repo teaches
what was implemented or studied
how correctness is checked
how performance is measured
what the next serious version would improve
```

## Step 1: Make The First Page Clear

The README should answer:

```text
what is this project?
who is it for?
what topics does it cover?
where should someone start?
what proves the work is real?
```

A reviewer should not need to guess whether the repo is:

```text
a course
a kernel library
a portfolio project
a research notebook
```

Say what it is directly.

## Step 2: Show The Learning Arc

The final capstone should make the sequence visible:

```text
GPU mental model
CPU references
CUDA basics
memory access
reductions
softmax and normalization
matmul tiling
Triton kernels
PyTorch integration
transformer kernels
attention and inference
portfolio packaging
```

This arc is the value of the project.

It shows that the course is not a random pile of kernels.

## Step 3: Point To Correctness Evidence

Correctness evidence includes:

```text
reference implementations
tests
baseline comparisons
test matrices
debugging notes
```

The final capstone should make this habit obvious:

```text
reference first, kernel second, benchmark third
```

That is the engineering spine of the repo.

## Step 4: Point To Performance Evidence

Performance evidence should be precise.

Good evidence names:

```text
operation
shape
dtype
baseline
timing method
hardware
result or observation
```

If a number is missing, do not fake it.

Write the benchmark structure and mark the result as pending.

Honest incomplete evidence is better than confident nonsense.

## Step 5: Separate Lessons From Results

The repo has two kinds of artifacts:

```text
weeks/   -> teaching and explanation
results/ -> notes, comparisons, and evidence
```

The capstone should preserve that separation.

Lessons teach the concept.

Results record what happened in the project.

Mixing them makes both harder to read.

## Step 6: Write The Final Project Summary

A final summary can be short:

```text
This project is a GPU kernel engineering roadmap that starts from CPU
references and simple CUDA kernels, then builds toward memory-aware matmul,
Triton kernels, PyTorch integration, transformer fusion patterns, attention
dataflow, FlashAttention concepts, and KV-cache inference tradeoffs.
```

Then add the proof:

```text
The repo emphasizes correctness through references and tests, and performance
through benchmark notes that name shapes, baselines, dtypes, and timing methods.
```

That is enough for the public story.

## Step 7: Name The Next Version

A finished project can still have a next version.

Good next steps are specific:

```text
add real GPU benchmark numbers across fixed hardware
implement one production-quality Triton attention kernel
add Nsight profiling screenshots or notes
expand PyTorch custom-op packaging
compare multiple tile sizes for matmul and attention
```

This tells the reader where the work would go next.

It also shows that you understand the difference between a learning repo and a
production kernel library.

## Final Capstone Shape

The final capstone should contain:

```text
one project summary
one topic map
one correctness story
one benchmark story
one attention/inference highlight
one resume-ready project paragraph
one next-version list
```

That is the whole course compressed into one readable artifact.

## The Core Pattern

A strong final capstone makes the project:

```text
easy to open
easy to navigate
easy to verify
easy to explain
easy to continue
```

The course is done when the repo can speak clearly without you standing next to
it.
