# Week 33: PyTorch Baselines

Weeks 25 through 32 taught Triton kernels, matmul tuning, batching, and
profiling.

Week 33 connects that work back to PyTorch.

The lesson is:

```text
before you trust a custom kernel, compare it to a PyTorch operation you trust
```

A baseline defines correctness.

Only after correctness is clear does performance mean anything.

## Step 1: Name The Operation

Start with one operation, not a whole model.

For vector add, the operation contract is:

```text
input:  x, y with the same shape
output: z with the same shape
rule:   z[i] = x[i] + y[i]
```

For matmul, the contract is:

```text
input:  A with shape [M, K]
input:  B with shape [K, N]
output: C with shape [M, N]
rule:   C[row, col] = dot(A[row, :], B[:, col])
```

The custom kernel can change how the work is done.

It cannot change what the operation means.

## Step 2: Write The PyTorch Version

The baseline should be boring.

That is its job.

```python
import torch


def torch_vector_add(x, y):
    return x + y


def torch_matmul(a, b):
    return a @ b
```

These functions are useful because they give the custom kernel something stable
to match.

## Step 3: Use The Same Inputs

The baseline and candidate must receive the same tensors.

This comparison is wrong:

```python
x1 = torch.randn(1024, device="cuda")
y1 = torch.randn(1024, device="cuda")
baseline = x1 + y1

x2 = torch.randn(1024, device="cuda")
y2 = torch.randn(1024, device="cuda")
candidate = custom_vector_add(x2, y2)
```

The outputs come from different random inputs.

The comparison should look like this:

```python
x = torch.randn(1024, device="cuda")
y = torch.randn(1024, device="cuda")

baseline = x + y
candidate = custom_vector_add(x, y)
```

One input set.

Two implementations.

One comparison.

## Step 4: Compare The Output Contract

Check the simple properties before values:

```python
assert candidate.shape == baseline.shape
assert candidate.dtype == baseline.dtype
assert candidate.device == baseline.device
```

These checks catch different classes of bugs:

```text
shape  -> indexing, allocation, grid size, edge handling
dtype  -> precision changes, wrapper mistakes, accumulator mistakes
device -> accidental CPU output or broken GPU integration
```

If any of these fail, stop there.

Do not benchmark a kernel that has not matched the output contract.

## Step 5: Compare Values With Tolerance

Floating point outputs usually should not be compared with exact equality.

Use a tolerance:

```python
torch.testing.assert_close(candidate, baseline, rtol=1e-4, atol=1e-4)
```

The right tolerance depends on dtype and operation.

For `float32`, small differences are normal.

For `float16` and `bfloat16`, slightly larger differences may be expected.

Choose the tolerance before claiming the kernel is correct.

Changing tolerance only after a failure can hide a real bug.

## Step 6: Put The Checks In One Helper

A small helper keeps every custom op honest in the same way:

```python
def compare_tensors(name, candidate, baseline, *, rtol=1e-4, atol=1e-4):
    assert candidate.shape == baseline.shape, f"{name}: shape mismatch"
    assert candidate.dtype == baseline.dtype, f"{name}: dtype mismatch"
    assert candidate.device == baseline.device, f"{name}: device mismatch"
    torch.testing.assert_close(candidate, baseline, rtol=rtol, atol=atol)
```

Read that helper as a debugging order:

```text
1. shape
2. dtype
3. device
4. values
```

If shape fails, inspect indexing and allocation.

If dtype fails, inspect the wrapper and accumulator path.

If values fail, inspect math, masks, boundaries, and precision.

## Step 7: Time Only After Correctness

CUDA work is asynchronous.

If you time without synchronization, you may only measure launch overhead.

A small timing helper can use CUDA events:

```python
def time_cuda(fn, *args, repeats=50):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        out = fn(*args)
    end.record()

    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / repeats
```

The sequence is:

```text
synchronize
record start
run repeated work
record end
synchronize
return output and average milliseconds
```

This is not a full benchmark suite, but it is honest enough for the first
baseline comparison.

## Step 8: Record The Shape

A useful result always names the conditions:

```text
operation: vector add
shape: [1_048_576]
dtype: float32
baseline: torch add
candidate: custom kernel
correct: yes
timing method: CUDA events, 50 repeats
```

Avoid this:

```text
my kernel is faster than PyTorch
```

Write this instead:

```text
for shape X, dtype Y, hardware Z, and timing method T, this candidate was faster
```

That is the difference between a useful benchmark and a vague claim.

## What The Baseline Protects

A PyTorch baseline catches problems before they turn into confusing performance
stories:

```text
wrong indexing
missing masks
bad edge handling
unexpected dtype changes
wrong output shape
precision errors
batch dimension mistakes
```

Most importantly, it stops you from optimizing code that computes the wrong
answer.

## The Core Pattern

For every custom kernel:

```text
define the operation contract
run the PyTorch baseline
run the custom candidate on the same inputs
compare shape, dtype, device, and values
benchmark only after correctness passes
record shape, dtype, hardware, baseline, and timing method
```

The custom kernel may be faster or more specialized.

It still has to behave like the operation it replaces.

## Bridge To Week 34

Week 34 teaches the wrapper boundary.

That is where the PyTorch baseline and custom kernel sit behind one clean
Python function, so the rest of the project can call the operation without
caring which backend is currently being tested.
