# Week 35: GPU Test Matrix

Week 33 taught PyTorch baselines.

Week 34 taught wrappers.

Week 35 teaches the test matrix: the set of cases that tells you whether a
custom operation is reliable across more than one easy input.

## Step 1: Start With The Contract

Every test matrix begins with the operation contract.

For vector add:

```text
x and y have the same shape
output has the same shape
output[i] = x[i] + y[i]
```

For matmul:

```text
A is [M, K]
B is [K, N]
C is [M, N]
```

The matrix should test that contract under different conditions.

## Step 2: Choose Shape Cases

Do not test only a perfect size like `1024`.

Kernels often fail at boundaries.

Useful vector-add shapes:

```text
1
31
32
33
1024
1025
```

These shapes test:

```text
tiny inputs
exact block boundaries
one element past a block boundary
larger inputs
```

For matmul, useful shapes include:

```text
M=1, K=64, N=64
M=64, K=64, N=64
M=65, K=64, N=64
M=128, K=96, N=80
```

The goal is to include both neat and awkward cases.

## Step 3: Choose Dtypes

Dtype changes correctness expectations.

Start with:

```text
float32
float16
```

For `float32`, tolerance can usually be tighter.

For `float16`, tolerance often needs to be looser.

A test can encode that:

```python
TOLERANCES = {
    torch.float32: dict(rtol=1e-4, atol=1e-4),
    torch.float16: dict(rtol=1e-2, atol=1e-2),
}
```

The test matrix should make dtype choices visible.

Hidden dtype assumptions become painful later.

## Step 4: Include Edge Cases

Edge cases are where masks and indexing bugs show up.

For elementwise kernels, include:

```text
empty or tiny tensors if supported
size smaller than block size
size equal to block size
size one larger than block size
non-power-of-two sizes
```

For row-wise kernels, include:

```text
one row
many rows
short rows
wide rows
width not divisible by block size
```

For batched kernels, include:

```text
batch size 1
batch size greater than 1
different M, N, and K values
```

The best test cases are boring to read and excellent at catching mistakes.

## Step 5: Write The Matrix As Data

A test matrix should be easy to scan.

Instead of copying the same test body many times, put cases in a list:

```python
VECTOR_ADD_CASES = [
    (1, torch.float32),
    (31, torch.float32),
    (32, torch.float32),
    (33, torch.float32),
    (1025, torch.float16),
]
```

Then one test can loop over the cases:

```python
for n, dtype in VECTOR_ADD_CASES:
    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.randn(n, device="cuda", dtype=dtype)
    baseline = x + y
    candidate = vector_add(x, y)
    torch.testing.assert_close(candidate, baseline, **TOLERANCES[dtype])
```

The loop is not fancy.

It is just a clean way to make the coverage visible.

## Step 6: Test Failures On Purpose

Correct inputs are only half the story.

Wrappers should reject invalid inputs clearly.

For vector add:

```python
with pytest.raises(ValueError):
    vector_add(torch.randn(8, device="cuda"), torch.randn(9, device="cuda"))
```

This confirms that the wrapper protects the kernel contract.

Without this test, invalid inputs may produce confusing kernel failures.

## Step 7: Keep Performance Out Of Correctness Tests

Correctness tests should answer:

```text
does the operation return the right result?
```

Benchmarks should answer:

```text
how fast is it?
```

Do not mix them too early.

Performance tests are usually slower, noisier, and more hardware-specific.

Correctness tests should be quick enough to run often.

## The Core Pattern

A GPU test matrix should cover:

```text
operation contract
normal shapes
boundary shapes
awkward shapes
dtypes
tolerances
invalid inputs
baseline comparison
```

The point is not to test every possible tensor.

The point is to choose cases that make common GPU bugs visible.

## Bridge To Week 36

Week 36 teaches debugging.

Once a test fails, you need a calm way to narrow the problem from "the kernel is
wrong" to a specific issue in shape, dtype, indexing, masking, math, or launch
configuration.
