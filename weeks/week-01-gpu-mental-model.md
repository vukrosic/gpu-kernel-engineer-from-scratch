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

## What You Need From The Repo

- [../course/recovery-system.md](../course/recovery-system.md)
- [../course/month-01-gpu-foundations.md](../course/month-01-gpu-foundations.md)
- [../gputriton/reference.py](../gputriton/reference.py)
- [../gputriton/bench.py](../gputriton/bench.py)
- [../tests/test_reference.py](../tests/test_reference.py)

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

## Write Down

Answer these in your note:

1. What is the simplest honest way to explain CPU vs GPU execution?
2. Why do reference implementations come before kernels?
3. What did the starter benchmark tell you that the tests did not?
4. What do you want to compare once GPU code exists?

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
