# Week 01: GPU Mental Model And Baseline

This week teaches the first habit of GPU kernel engineering:

```text
trust the reference before you trust the GPU
```

You will not write CUDA yet. You will learn the CPU-vs-GPU mental model, run
the starter reference code, understand what the tests prove, understand what the
benchmark measures, and write your first baseline note.

Work through this file from top to bottom. Every task appears inside the lesson
at the moment you need it.

## Step 1: Understand The Job Of Week 01

The goal this week is not to make anything fast. The goal is to learn how to
tell whether later GPU code is correct.

The workflow is:

```text
1. understand the operation
2. run a simple trusted implementation
3. test that the implementation returns the right answer
4. measure how long the implementation takes
5. write down what happened
```

For Week 01, the trusted implementation is CPU/NumPy code. NumPy is not the
final destination of the course. It is just the baseline that tells you what the
answer should be before you write GPU code.

## Step 2: Learn The CPU vs GPU Mental Model

A CPU is built for flexible control flow and fast response to individual tasks.
It is good at running Python code, operating-system work, branching logic,
database queries, web requests, and programs where each next step may depend on
the previous step.

A GPU is built for throughput. It is good at doing a huge number of similar
operations over large arrays of data. Instead of imagining one flexible worker
doing the whole job, imagine many simple workers each handling one piece.

The simplest example is vector add:

```text
c[i] = a[i] + b[i]
```

If the arrays are:

```text
a = [1, 2, 3]
b = [4, 5, 6]
```

then the result is:

```text
c = [5, 7, 9]
```

On a CPU, you can imagine a loop:

```python
for i in range(n):
    c[i] = a[i] + b[i]
```

On a GPU, you try to give many workers one output element each:

```text
worker 0 computes c[0]
worker 1 computes c[1]
worker 2 computes c[2]
...
```

Vector add fits the GPU mental model because each output element is independent.
To compute `c[2]`, you only need `a[2]` and `b[2]`. You do not need to know
what happened at `c[0]` or `c[1]`.

This is the first useful GPU question:

```text
Can many output elements be computed independently?
```

If the answer is yes, the problem may fit a GPU. If the answer is no, the GPU
may still help, but the kernel will need more careful design.

## Step 3: Learn The Words Host, Device, And Kernel

GPU programming uses three words constantly:

- **host** means the CPU side of the program
- **device** means the GPU side of the program
- **kernel** means a function launched to run on the GPU

For now, keep the model simple:

```text
host code:
    prepares input arrays
    launches GPU work
    waits for GPU work to finish
    checks the output

device code:
    runs many workers in parallel
    gives each worker a piece of the data
```

You are not writing device code this week. You are building the reference and
baseline habit that will make device code trustworthy next week.

## Step 4: Run The Project Setup

From the repo root, install the project in editable mode with its development
dependencies:

```bash
python -m pip install -e ".[dev]"
```

This command makes the local package importable and installs the tools needed
for the starter tests.

If this command fails, stop and record the error in `results/week-01-baseline.md`
later. Environment failures are part of systems work. Do not pretend they did
not happen.

## Step 5: Run The First Correctness Check

Now run:

```bash
pytest
```

This does not prove that the course is complete. It proves that the current
reference functions satisfy the starter correctness checks.

One of the tests checks vector add:

```python
def test_vector_add():
    assert np.allclose(
        vector_add(np.array([1, 2]), np.array([3, 4])),
        np.array([4, 6]),
    )
```

Read the test as a sentence:

```text
When vector_add receives [1, 2] and [3, 4],
it should return [4, 6].
```

That is what a correctness test does. It gives an input, names the expected
output, and fails if the actual output does not match.

If `pytest` passes, write down that it passed. If it fails, write down the
failure. The result note should describe reality, not the ideal path.

## Step 6: Inspect The Reference Functions

The reference functions live in `gputriton/reference.py`. You can inspect the
important ones directly:

```bash
python - <<'PY'
import inspect
from gputriton import reference

print(inspect.getsource(reference.vector_add))
print(inspect.getsource(reference.softmax))
PY
```

The vector-add reference is intentionally simple:

```python
def vector_add(a, b):
    return np.asarray(a) + np.asarray(b)
```

It converts the inputs to NumPy arrays and adds them element by element. This is
not a GPU kernel. It is the trusted answer for future GPU kernels.

The softmax reference is a little more interesting:

```python
def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)
```

Do not worry about implementing softmax yet. For Week 01, notice two habits:

1. the reference is readable
2. the reference is written for correctness before speed

That is the point. A reference implementation should be boring, obvious, and
easy to compare against.

Now write one sentence for each function you inspected:

```text
vector_add:
softmax:
```

Example:

```text
vector_add: adds matching positions from two arrays.
softmax: converts a row of scores into normalized probabilities.
```

## Step 7: Do The Week 01 Task

Your task is to produce one short baseline note:

```text
results/week-01-baseline.md
```

Run this command:

```bash
python examples/reference_bench.py
```

Then update the note with this structure:

```markdown
# Week 01 Baseline

## Environment

- Machine:
- OS:
- Python:
- NumPy:
- GPU, if any:

## Commands I Ran

- `python -m pip install -e ".[dev]"`
- `pytest`
- `python examples/reference_bench.py`

## Correctness Result

Record whether `pytest` passed. If it failed, paste the important error.

## Reference Timings

Paste the benchmark output here.

## Reference Functions

- `vector_add`:
- `softmax`:

## My CPU vs GPU Mental Model

Write 3-5 sentences in your own words.

## Test vs Benchmark

Answer these two questions:

- What does `pytest` prove?
- What does `python examples/reference_bench.py` measure?

## One Thing Still Fuzzy

Write one concept you want to understand better.
```

Keep it short. If you post in Skool, use this format:

```text
Week 01 Submission

Correctness result:

Benchmark result:

My explanation of CPU vs GPU:

Why references come before kernels:

One thing I want reviewed:
```

## Done Checklist

You are done when:

- `pytest` passed, or you recorded the error
- `python examples/reference_bench.py` ran, or you recorded the error
- `results/week-01-baseline.md` has your benchmark output
- your note explains CPU vs GPU in plain language
- your note names one thing you still do not understand

Stop there.
