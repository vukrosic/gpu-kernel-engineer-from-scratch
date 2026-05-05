# Week 25: Triton Mental Model

Triton is a Python-based way to write GPU kernels.

It does not remove the GPU ideas you learned.

It gives you a different way to express them.

CUDA often makes you think:

```text
threads, blocks, shared memory
```

Triton often makes you think:

```text
program instances, blocks of data, masks
```

Week 25 teaches that mental shift.

## Program Instances

A Triton kernel runs many program instances.

You can think of one program instance as:

```text
one tile of work
```

For a vector with 1000 elements and a block size of 256:

```text
program 0 handles elements 0-255
program 1 handles elements 256-511
program 2 handles elements 512-767
program 3 handles elements 768-1023
```

The last program runs past the end.

That is why masks matter.

## Program IDs

In Triton, a program finds its tile with a program id:

```python
pid = tl.program_id(0)
```

For a 1D vector:

```text
pid 0 -> first block
pid 1 -> second block
pid 2 -> third block
```

The start offset is:

```python
start = pid * BLOCK_SIZE
```

Then the program builds offsets:

```python
offsets = start + tl.arange(0, BLOCK_SIZE)
```

Those offsets represent a whole block of elements.

## Blocks Of Data

In CUDA, you might imagine many individual threads.

In Triton, you often write operations over a block of values:

```python
x = tl.load(x_ptr + offsets, mask=mask)
```

`x` is not one scalar.

It is a vector of values for this program instance.

That is the key shift:

```text
write block operations
let Triton lower them to GPU execution
```

You still need correct indexing and masks.

## Masks

A mask says which offsets are valid:

```python
mask = offsets < n
```

For `n = 10` and `BLOCK_SIZE = 4`:

```text
program 0 offsets: [0, 1, 2, 3] mask: [T, T, T, T]
program 1 offsets: [4, 5, 6, 7] mask: [T, T, T, T]
program 2 offsets: [8, 9, 10, 11] mask: [T, T, F, F]
```

The mask prevents invalid loads and stores.

Boundary handling is not a side detail in Triton.

It is part of the normal kernel shape.

## Triton Vector Add Shape

A vector add kernel has this structure:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y

    tl.store(out_ptr + offsets, out, mask=mask)
```

Read it in five parts:

```text
choose program id
build offsets
build mask
load block values
store masked results
```

That pattern repeats constantly.

## CUDA Vs Triton

CUDA elementwise thinking:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    out[i] = x[i] + y[i];
}
```

Triton elementwise thinking:

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n
out = tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask)
tl.store(out_ptr + offsets, out, mask=mask)
```

CUDA makes the scalar thread explicit.

Triton makes the block of offsets explicit.

Both need the same correctness idea:

```text
do not read or write outside the valid range
```

## What Triton Does Not Hide

Triton does not remove:

```text
memory layout
coalescing
row and column indexing
tile size choices
masks
benchmarking
```

It helps express kernels more compactly.

It does not make performance automatic.

You still need GPU engineering judgment.

## The Core Pattern

When reading a Triton kernel, ask:

```text
What does one program instance own?
What are the offsets?
What is the mask?
What values are loaded as blocks?
What is stored?
Which dimensions are constexpr block sizes?
```

If you can answer those, Triton kernels become much less mysterious.

## Bridge To Week 26

Week 26 applies this model to vector add and masks directly.

The next lesson is about edge blocks:

```text
what happens when the data size is not a clean multiple of the block size?
```
