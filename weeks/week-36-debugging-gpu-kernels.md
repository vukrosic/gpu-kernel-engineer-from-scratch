# Week 36: Debugging GPU Kernels

By Week 36, you have baselines, wrappers, and a test matrix.

Now assume one case fails.

This lesson teaches a debugging order for GPU kernels.

## Step 1: Reproduce One Failing Case

Do not debug the whole test suite at once.

Pick one failing case and freeze it:

```text
operation: vector add
shape: 1025
dtype: float32
device: cuda
failure: values differ at the last element
```

That note already gives you a direction.

If the last element fails, look at masking and edge handling before rewriting
the whole kernel.

## Step 2: Check The Wrapper

Before reading kernel math, inspect the wrapper.

Ask:

```text
is the output allocated with the right shape?
is the output allocated with the right dtype?
is the input length passed correctly?
is the launch grid large enough?
```

For a one-dimensional kernel, a common grid calculation is:

```python
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
```

If this uses floor division by accident, the final partial block will not run.

That is a wrapper bug, not a math bug.

## Step 3: Compare Against A Tiny Case

Tiny cases make indexing visible.

For vector add:

```text
x = [10, 20, 30]
y = [1,  2,  3]
z = [11, 22, 33]
```

For matmul:

```text
A = [[1, 2]]
B = [[3],
     [4]]
C = [[11]]
```

If the tiny case fails, the problem is probably basic indexing or operation
logic.

If tiny cases pass but larger awkward cases fail, suspect masks, boundaries, or
tiling.

## Step 4: Inspect Masks

Many GPU bugs live at the edge of the tensor.

In Triton, a masked load often looks like:

```python
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x_values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

The store needs a mask too:

```python
tl.store(out_ptr + offsets, result, mask=mask)
```

A masked load without a masked store can still write outside the valid output
range.

A masked store without a correct load can store bad values.

Check both sides.

## Step 5: Separate Shape Bugs From Value Bugs

Debug in this order:

```text
shape
dtype
device
values
performance
```

If shape is wrong, inspect allocation and grid mapping.

If dtype is wrong, inspect wrapper conversion and accumulator dtype.

If values are wrong, inspect indexing, masks, math, and precision.

If only performance is bad, inspect memory access, occupancy, tile size, and
profiling output.

The order matters because each failure type points to different code.

## Step 6: Reduce The Kernel Mentally

When a kernel is confusing, rewrite its behavior as plain indexing.

For elementwise:

```text
program id chooses a block
offsets choose positions inside the block
mask keeps valid offsets
load x and y
compute x + y
store result
```

For matmul:

```text
program id chooses a C tile
offsets choose rows and columns
K loop loads A and B tiles
accumulator holds partial sums
mask protects edges
store writes C tile
```

This mental reduction removes launch syntax and leaves the data movement.

## Step 7: Change One Thing

Once you have a hypothesis, change one thing.

Good debugging changes are small:

```text
try one smaller shape
print or record one shape value
replace custom output with baseline output temporarily
tighten one mask
change one grid formula
test one dtype
```

Avoid making several fixes at once.

If the test passes afterward, you will not know which change mattered.

## The Core Pattern

When a GPU test fails:

```text
freeze one failing case
check wrapper shape, dtype, device, and grid
try a tiny input
inspect masks and edge handling
separate shape bugs from value bugs
reduce the kernel to plain indexing
change one thing
rerun the same case
```

Debugging kernels is mostly disciplined narrowing.

The goal is not to be clever.

The goal is to make the bug smaller until it has only one place to hide.

## Bridge To Week 37

Week 37 moves back into transformer-style kernels with activation fusion.

The same baseline, wrapper, test, and debugging habits now apply to operations
that appear inside real neural network blocks.
