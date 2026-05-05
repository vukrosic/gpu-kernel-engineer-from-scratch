# Week 34: Custom Op Wrapper

Week 33 gave you the baseline rule:

```text
compare the custom path to a trusted PyTorch path
```

Week 34 teaches the boundary between normal Python code and a custom kernel.

That boundary is the wrapper.

## Step 1: Understand What A Wrapper Does

A wrapper is the Python function the rest of the project calls.

It should make the custom kernel feel like a normal PyTorch operation:

```python
z = vector_add(x, y)
```

The caller should not need to know about:

```text
launch grids
block sizes
Triton meta-parameters
temporary output allocation
kernel function names
```

Those details belong behind the wrapper.

## Step 2: Keep The Wrapper Small

A wrapper should not become a second implementation of the operation.

For vector add, the wrapper shape is:

```python
def vector_add(x, y):
    out = torch.empty_like(x)
    grid = compute_grid(x.numel())
    vector_add_kernel[grid](x, y, out, x.numel())
    return out
```

Read it in order:

```text
allocate output
choose launch grid
call kernel
return output
```

The wrapper prepares the call.

The kernel does the parallel work.

## Step 3: Validate Inputs Before Launch

The wrapper is the right place for simple input checks.

For vector add:

```python
def check_vector_add_inputs(x, y):
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    if x.dtype != y.dtype:
        raise ValueError("x and y must have the same dtype")
```

These checks protect the kernel from impossible assumptions.

A kernel is usually written for a specific contract.

The wrapper should reject inputs that break that contract.

## Step 4: Separate Reference And Kernel Paths

A useful wrapper can expose a reference path while development is still moving:

```python
def vector_add(x, y, *, backend="kernel"):
    check_vector_add_inputs(x, y)

    if backend == "reference":
        return x + y
    if backend != "kernel":
        raise ValueError(f"unknown backend: {backend}")

    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](x, y, out, x.numel())
    return out
```

Now the same public function can run:

```text
PyTorch reference
custom kernel
```

That makes tests and debugging cleaner.

## Step 5: Do Not Hide Too Much

The wrapper should hide launch mechanics.

It should not hide correctness problems.

Avoid wrappers that silently fix inputs:

```python
# Bad idea for early kernel work.
x = x.contiguous().float().cuda()
```

That kind of conversion can hide layout, dtype, and device bugs.

In early GPU engineering, prefer explicit failure:

```python
if not x.is_cuda:
    raise ValueError("x must be a CUDA tensor")
```

Later, production code may choose more flexible behavior.

The learning version should stay honest.

## Step 6: Match PyTorch Calling Style

Good wrappers feel familiar:

```python
out = custom_matmul(a, b)
```

For optional controls, use keyword arguments:

```python
out = custom_matmul(a, b, backend="triton")
```

Avoid APIs that expose too many kernel details to normal callers:

```python
out = custom_matmul(a, b, block_m=16, block_n=32, warps=4, stages=3)
```

Those knobs are useful for tuning experiments.

They should not be required for the basic call.

## Step 7: Test The Wrapper, Not Only The Kernel

The wrapper can have bugs even when the kernel is correct.

Wrapper bugs include:

```text
wrong output shape
wrong dtype allocation
wrong grid calculation
forgotten input validation
wrong backend path
```

A wrapper test should compare it to the PyTorch baseline:

```python
x = torch.randn(1024, device="cuda")
y = torch.randn(1024, device="cuda")

baseline = x + y
candidate = vector_add(x, y)

torch.testing.assert_close(candidate, baseline)
```

This checks the user-facing function, not only the low-level kernel.

## The Core Pattern

When writing a custom op wrapper:

```text
define the public function
validate inputs
allocate output
compute launch grid
call the kernel
return a normal PyTorch tensor
compare the wrapper output to a PyTorch baseline
```

The wrapper is successful when the rest of the code can call the custom op like
ordinary PyTorch.

## Bridge To Week 35

Week 35 turns this into a testing system.

Once you have a baseline and a wrapper, the next question is which shapes,
dtypes, and edge cases must be checked before you trust the operation.
