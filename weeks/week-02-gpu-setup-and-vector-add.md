# Week 02: GPU Setup And Vector Add

## What This Week Is

Week 02 is the first hands-on step after the mental model. You turn the idea of
a kernel into a tiny, testable vector-add workflow and compare it to the
CPU/NumPy baseline already in the repo.

This is still a small problem on purpose. Vector add is the first place where
you can see the full loop of this course: write the shape, check the result, and
then think about how a GPU would carry the same work.

## What You Need From The Repo

- [../gputriton/reference.py](../gputriton/reference.py)
- [../gputriton/bench.py](../gputriton/bench.py)
- [../tests/test_reference.py](../tests/test_reference.py)
- [../course/month-01-gpu-foundations.md](../course/month-01-gpu-foundations.md)
- [../cuda/vector_add.cu](../cuda/vector_add.cu)
- [../triton_kernels/vector_add.py](../triton_kernels/vector_add.py)

## Exact Commands

```bash
python -m pip install -e ".[dev]"
pytest
python examples/reference_bench.py
```

Run a tiny vector-add experiment and compare it to the reference:

```bash
python - <<'PY'
import numpy as np
from gputriton.reference import vector_add

a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
print(vector_add(a, b))
PY
```

## Build This

Create `results/week-02-vector-add.md` and use it to explain:

- what vector add does
- why it is a good first kernel shape
- what would change if the same logic ran on a GPU
- what you would check first if the result looked wrong

Then write a short kernel sketch in the same file:

- one input element per thread
- one output element per thread
- one memory read from `a`
- one memory read from `b`
- one memory write to the output

If you have a CUDA environment, you can also sketch the same idea in
`cuda/vector_add.cu` or `triton_kernels/vector_add.py`, but the week is still
fully valid if you stay on the reference path.

## Code Sketch

```python
def vector_add(a, b):
    out = []
    for x, y in zip(a, b):
        out.append(x + y)
    return out
```

This sketch is correct because every output element is the sum of one matching
element from `a` and one matching element from `b`.

## Write Down

Answer these in your note:

1. Why is vector add the simplest useful GPU kernel?
2. Why do we compare against a CPU/NumPy reference first?
3. What does the output shape tell you about the work being done?
4. What would a GPU version change about the memory access pattern?

## Minimum

- `pytest` passes
- `python examples/reference_bench.py` runs
- `results/week-02-vector-add.md` exists
- you explain vector add in your own words

## Standard

- you compare at least two input sizes
- you explain the one-thread-one-element mapping
- you write one sentence about why vector add is memory-light but concept-heavy

## Stretch

- you write a CUDA-style pseudocode version of vector add
- you compare the vector-add benchmark to the matmul benchmark from Week 01
- you explain why "easy math" does not mean "easy GPU performance"

## If You Are Behind

Do only the Minimum path. Do not skip the benchmark run, and do not skip the
results note.

## Next Week

Week 03 turns the same idea into indexing: grids, blocks, threads, and
elementwise kernels.
