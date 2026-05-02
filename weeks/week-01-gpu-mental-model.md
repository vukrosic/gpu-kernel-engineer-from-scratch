# Week 01: GPU Mental Model And Baseline

## What This Week Is

This is the first week of the year-long course, and it starts with the most
important habit in the whole project: trust the reference before you trust the
GPU.

You are not trying to get clever yet. You are building the workflow that keeps
later kernels honest:

1. understand the shape of the problem
2. run the baseline code
3. verify correctness
4. record an observation
5. write down what changed in your own words

The week uses CPU and NumPy reference code on purpose. Before any CUDA or Triton
work matters, you need a baseline that future kernels can be compared against.

## Lesson: CPU Work, GPU Work, And Baselines

Read this section before running commands. It contains the mental model you need
for the rest of the week.

### CPU vs GPU

A CPU is built for flexible control flow and fast response to individual tasks.
It is good at running operating-system work, Python code, branching logic,
database queries, web requests, and all the messy jobs where the next step may
depend heavily on the previous step.

A GPU is built for throughput. It is good at doing a huge number of similar
operations over large arrays of data. Instead of thinking "one smart worker
does the whole job," think "many simple workers each handle a small piece."

That difference matters because a GPU is not automatically faster. Moving work
to the GPU has overhead. Starting GPU work has overhead. Reading and writing GPU
memory has costs. The GPU wins when there is enough parallel work to make those
costs worth paying.

### Host, Device, And Kernel

In GPU programming, you will see these words constantly:

- **host** means the CPU side of the program
- **device** means the GPU side of the program
- **kernel** means a function launched to run on the GPU

The host prepares the inputs, launches the kernel, and checks the output. The
device runs many copies of the kernel work in parallel.

For now, keep the model simple:

```text
CPU host code:
    prepare input arrays
    launch GPU kernel
    wait for GPU work to finish
    check output against reference

GPU device code:
    many workers run the same kernel
    each worker handles one piece of the data
```

### Why Vector Add Is The First Kernel

Think of vector add:

```text
c[i] = a[i] + b[i]
```

On a CPU, you can imagine one loop walking through the arrays:

```python
for i in range(n):
    c[i] = a[i] + b[i]
```

On a GPU, you try to give many workers one small job each:

```text
thread 0 computes c[0]
thread 1 computes c[1]
thread 2 computes c[2]
...
```

This is a good first GPU example because every output position is independent.
To compute `c[17]`, you only need `a[17]` and `b[17]`. You do not need to know
what happened at `c[16]` or `c[18]`.

That independence is the first signal that a problem might fit the GPU.

When you see an operation, ask:

1. Can many output elements be computed independently?
2. Does each worker do similar work?
3. Is there enough data to make parallel execution worthwhile?
4. Can I write a simple CPU reference for the same result?

Vector add says yes to all four.

### Why The Course Starts Without CUDA

This week uses CPU and NumPy code even though the course is about GPU kernels.
That is intentional.

Before you write a GPU kernel, you need a trusted answer. The easiest trusted
answer is usually a plain CPU or NumPy implementation:

```python
def vector_add(a, b):
    return a + b
```

Later, when you write a CUDA or Triton version, the question becomes:

```text
Does my GPU output match the trusted CPU/NumPy output?
```

If the answer is no, speed does not matter. A fast wrong kernel is still wrong.

### Reference, Test, And Benchmark

These three words are separate on purpose:

- **reference**: the simple trusted implementation
- **test**: the correctness check that compares actual output to expected output
- **benchmark**: the timing measurement that shows how long something took

A test answers:

```text
Did this produce the right result?
```

A benchmark answers:

```text
How long did this take on this machine, with this input, measured this way?
```

Both are necessary. Tests keep you honest about correctness. Benchmarks keep you
honest about performance. Neither replaces the other.

### How To Read Benchmark Numbers

Week 01 benchmark numbers are not final performance claims. They are a baseline
for your own machine.

When you record a benchmark, always include context:

- what command you ran
- what machine you used
- what Python and library versions you used, if available
- what input size the benchmark used
- whether you ran it once or multiple times

Timing can vary. That does not make benchmarking useless. It means benchmark
numbers are evidence, not magic facts. Later in the course, you will improve the
timing harness with warmups, repeats, synchronization, and GPU-specific timers.

For Week 01, the goal is simpler: run the starter benchmark, record what it
prints, and explain what kind of question the benchmark answers.

### What You Should Understand Before The Tasks

By the time you finish this lesson section, you should be able to say:

- a CPU is usually better for flexible sequential control
- a GPU is usually better for large parallel array work
- host means CPU side
- device means GPU side
- a kernel is GPU-side work launched by the host
- vector add is parallel because each output element is independent
- a reference implementation defines the trusted answer
- a test checks correctness
- a benchmark measures time
- correctness comes before speed

This is why the course starts with baselines. Before asking whether a GPU kernel
is fast, you need to know what answer it should produce and what simple CPU-side
implementation you are comparing against.

## What You Need From The Repo

After the lesson above, inspect the repo files that support the workflow. These
files are not extra reading for the GPU mental model. They show how this project
turns the mental model into code, tests, and benchmark output.

- [../course/recovery-system.md](../course/recovery-system.md)
- [../course/month-01-gpu-foundations.md](../course/month-01-gpu-foundations.md)
- [../gputriton/reference.py](../gputriton/reference.py)
- [../gputriton/bench.py](../gputriton/bench.py)
- [../tests/test_reference.py](../tests/test_reference.py)

Read them in this order:

1. `course/month-01-gpu-foundations.md` to see why Week 01 exists.
2. `gputriton/reference.py` to see the CPU/NumPy truth sources.
3. `tests/test_reference.py` to see what correctness means in this repo.
4. `gputriton/bench.py` and `examples/reference_bench.py` to see how the first
   timings are produced.

While reading, write one sentence per file:

- What does this file teach?
- What would break later in the course if this file were wrong?

## Exact Commands

From the repo root, set up the dev environment and run the checks:

```bash
python -m pip install -e ".[dev]"
pytest
python examples/reference_bench.py
```

Then inspect the reference implementations directly:

```bash
python - <<'PY'
import inspect
from gputriton import reference

print(inspect.getsource(reference.vector_add))
print(inspect.getsource(reference.softmax))
PY
```

## Build This

Create `results/week-01-baseline.md` and turn it into a real baseline note. It
should capture:

- the commands you ran
- the environment you ran them in
- the reference timings you saw
- your first plain-language CPU vs GPU explanation
- one sentence that previews what Week 02 will test next

Use this outline:

```markdown
# Week 01 Baseline

## Environment

- Machine:
- OS:
- Python:
- NumPy:
- GPU, if any:

## Commands

- `python -m pip install -e ".[dev]"`
- `pytest`
- `python examples/reference_bench.py`

## Reference Timings

Paste the benchmark output here.

## What I Learned

Explain CPU vs GPU execution in your own words.

## Reference Functions

- `vector_add`:
- `matmul`:
- `softmax`:
- `attention`:

## Questions

Answer the Week 01 questions here.

## Next Week

What do you expect the first GPU vector-add kernel to prove?
```

## Code Sketch

```python
import time

def measure(run):
    start = time.perf_counter()
    result = run()
    elapsed = time.perf_counter() - start
    return result, elapsed
```

This sketch is correct because it keeps the work being measured separate from
the act of measuring it. That separation is the first step toward trustworthy
benchmarks.

You are not expected to add this function to the repo yet. It is here so you
can see the shape of a timing helper: pass in work, measure it, return both the
result and the elapsed time. Later weeks will make this stricter with warmups,
repeats, synchronization, and GPU-aware timing.

## Write Down

Answer these in your note:

1. What is the simplest honest way to explain CPU vs GPU execution?
2. Why do reference implementations come before kernels?
3. What did the starter benchmark tell you that the tests did not?
4. What do you want to compare once GPU code exists?
5. Which concept from this week is still unclear?

Use the lesson and repo files this way:

- Question 1 comes from the CPU vs GPU and vector-add sections.
- Question 2 comes from `gputriton/reference.py` and `tests/test_reference.py`.
- Question 3 comes from comparing `pytest` with `python examples/reference_bench.py`.
- Question 4 comes from the Month 01 roadmap.
- Question 5 is your signal for what to ask in the community, office hours, or
  your own follow-up study.

## Minimum

- `pytest` runs
- `python examples/reference_bench.py` runs
- `results/week-01-baseline.md` exists
- you can explain the CPU vs GPU mental model in plain language

## Standard

- you record the reference outputs or timings from the benchmark run
- you write a short paragraph about why baselines matter
- you summarize the reference code in one sentence per function

## Stretch

- you compare two different input sizes or shapes
- you write a short "course promise" paragraph in your own voice
- you note one thing you still do not understand about GPU work

## If You Are Behind

Do the install, the tests, and the benchmark. Then write five sentences about
why baselines matter and stop there.

## Next Week

Week 02 turns the mental model into the first kernel-shaped exercise: vector
add, one output per input position, checked against the CPU reference.

## Done Checklist

Minimum:

- [ ] `pytest` passes
- [ ] `python examples/reference_bench.py` runs
- [ ] You wrote five sentences explaining CPU vs GPU

Standard:

- [ ] `results/week-01-baseline.md` exists
- [ ] It includes benchmark numbers
- [ ] It includes one-sentence explanations of the reference functions
- [ ] It includes answers to the five mental-model questions
- [ ] It names one concept that now makes sense and one concept that is still fuzzy

Stretch:

- [ ] You ran the benchmark three times and compared variation
- [ ] You changed the input sizes in `gputriton/bench.py` and observed what changed
- [ ] You wrote one question you want Week 02 to answer

## What To Skip If Overwhelmed

Skip Stretch.

Skip changing benchmark input sizes.

Do not skip tests. Do not skip the baseline benchmark. Those are the habits the
whole course depends on.

## What Week 02 Will Do

Week 02 starts the first real GPU kernel path:

- CUDA setup
- vector add as the first kernel
- CPU/NumPy reference comparison
- first GPU benchmark

The Week 01 baseline is what lets you know whether the Week 02 kernel is correct.
