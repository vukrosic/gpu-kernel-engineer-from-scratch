# Week 06: Coalescing Vs Strides

Week 05 taught you to ask:

```text
How many bytes does this kernel move?
```

Week 06 adds the next question:

```text
Are nearby threads reading nearby memory?
```

Two kernels can move the same number of bytes and still behave differently. The
difference often comes from memory access pattern.

The key idea is:

```text
GPUs like groups of threads to access contiguous memory.
```

That is the beginning of coalescing.

## Contiguous Memory

Contiguous memory means values sit next to each other in memory.

For a 1D array:

```text
x = [10, 20, 30, 40, 50, 60, 70, 80]
```

The flat memory layout is:

```text
index:  0   1   2   3   4   5   6   7
value: 10  20  30  40  50  60  70  80
```

If consecutive threads read consecutive elements:

```text
thread 0 reads x[0]
thread 1 reads x[1]
thread 2 reads x[2]
thread 3 reads x[3]
```

Then the threads are reading nearby memory.

This is the friendly case.

## Strided Memory

A stride is the step between accessed elements.

Stride `1` means contiguous access:

```text
x[0], x[1], x[2], x[3], x[4]
```

Stride `2` means every other element:

```text
x[0], x[2], x[4], x[6], x[8]
```

Stride `4` means:

```text
x[0], x[4], x[8], x[12], x[16]
```

The larger the stride, the more spread out the memory access becomes.

In kernel-shaped code, contiguous access looks like:

```cpp
float value = x[i];
```

Strided access looks like:

```cpp
float value = x[i * stride];
```

Both can be correct.

They are not equally friendly to hardware.

## Why Nearby Access Matters

The GPU does not usually fetch one tiny value in total isolation.

Memory is moved in chunks.

If a group of nearby threads asks for nearby addresses, the hardware can serve
that access efficiently.

If those threads ask for scattered addresses, the hardware may need more memory
transactions to gather the same number of useful values.

The simple mental model is:

```text
nearby threads + nearby addresses = easier memory access
nearby threads + scattered addresses = harder memory access
```

That is why access pattern matters even when the math is identical.

## Warps

CUDA threads execute in groups called warps.

On NVIDIA GPUs, a warp is commonly:

```text
32 threads
```

You can think of a warp as a small pack of threads that execute together.

If a warp reads contiguous `float32` values:

```text
thread 0  -> x[0]
thread 1  -> x[1]
thread 2  -> x[2]
...
thread 31 -> x[31]
```

Those 32 floats occupy:

```text
32 * 4 bytes = 128 bytes
```

That is a clean, compact memory region.

This is the pattern GPUs are built to like.

## Coalesced Access

Coalesced access means the memory requests from nearby threads can be combined
into efficient memory transactions.

The beginner version is:

```text
consecutive threads should usually read consecutive memory addresses
```

Coalesced:

```text
thread 0 -> x[0]
thread 1 -> x[1]
thread 2 -> x[2]
thread 3 -> x[3]
```

Not coalesced:

```text
thread 0 -> x[0]
thread 1 -> x[1000]
thread 2 -> x[2000]
thread 3 -> x[3000]
```

Both patterns read four values.

The second pattern spreads those reads across memory.

That usually means more memory work for the hardware.

## A Coalesced Copy Kernel

This is the copy pattern from Week 05:

```cpp
__global__ void copy_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = x[i];
    }
}
```

For consecutive threads:

```text
thread 0 -> x[0] -> out[0]
thread 1 -> x[1] -> out[1]
thread 2 -> x[2] -> out[2]
thread 3 -> x[3] -> out[3]
```

Reads are contiguous.

Writes are contiguous.

This is the simplest coalesced elementwise pattern.

## A Strided Copy Kernel

Now compare that to a strided copy:

```cpp
__global__ void strided_copy_kernel(
    const float* x,
    float* out,
    int n,
    int stride
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = x[i * stride];
    }
}
```

For `stride = 4`:

```text
thread 0 -> x[0]
thread 1 -> x[4]
thread 2 -> x[8]
thread 3 -> x[12]
```

Writes are still contiguous:

```text
out[0], out[1], out[2], out[3]
```

But reads are spread out:

```text
x[0], x[4], x[8], x[12]
```

The kernel is not automatically wrong.

It just asks memory for values in a less convenient pattern.

## Rows Are Usually Friendly

For a row-major matrix:

```text
[
  [ 0,  1,  2,  3],
  [ 4,  5,  6,  7],
  [ 8,  9, 10, 11],
]
```

Rows are contiguous in memory:

```text
row 0 -> 0, 1, 2, 3
row 1 -> 4, 5, 6, 7
row 2 -> 8, 9, 10, 11
```

If consecutive threads walk across a row:

```text
thread 0 -> matrix[0, 0]
thread 1 -> matrix[0, 1]
thread 2 -> matrix[0, 2]
thread 3 -> matrix[0, 3]
```

The flat addresses are:

```text
0, 1, 2, 3
```

That is contiguous.

## Columns Are Usually Strided

In the same row-major matrix, a column is not contiguous:

```text
matrix[0, 1] -> flat index 1
matrix[1, 1] -> flat index 5
matrix[2, 1] -> flat index 9
```

The step between rows is the matrix width:

```text
stride = width
```

So if consecutive threads walk down a column:

```text
thread 0 -> matrix[0, 1] -> flat index 1
thread 1 -> matrix[1, 1] -> flat index 5
thread 2 -> matrix[2, 1] -> flat index 9
```

The addresses jump:

```text
1, 5, 9
```

This is strided access.

That is why row-wise and column-wise kernels can behave differently on
row-major data.

## Same Math, Different Access

Imagine two kernels that both add `1.0` to every element of a matrix.

Row-friendly access:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int i = row * width + col;

out[i] = x[i] + 1.0f;
```

If neighboring threads differ mostly in `col`, they touch nearby addresses.

Column-strided access:

```cpp
int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;
int i = row * width + col;

out[i] = x[i] + 1.0f;
```

Now neighboring threads may differ mostly in `row`.

If `width` is large, those addresses are far apart.

The operation is still:

```cpp
out[i] = x[i] + 1.0f;
```

But the memory pattern changed.

## Correct Does Not Mean Fast

A strided kernel can produce the right answer.

That does not mean it uses the hardware well.

This distinction matters:

```text
correctness asks: did we compute the right values?
performance asks: did we access memory in a hardware-friendly way?
```

Early in the course, correctness comes first.

Once correctness is stable, memory layout becomes one of the first performance
questions to ask.

## Coalescing And Stores

Coalescing matters for writes too.

Friendly writes:

```text
thread 0 -> out[0]
thread 1 -> out[1]
thread 2 -> out[2]
thread 3 -> out[3]
```

Scattered writes:

```text
thread 0 -> out[0]
thread 1 -> out[1000]
thread 2 -> out[2000]
thread 3 -> out[3000]
```

Scattered writes are especially important to notice because they can also create
correctness risks if two threads accidentally write the same location.

For simple elementwise kernels, the clean pattern is:

```text
contiguous reads
contiguous writes
one thread writes one output element
```

## Strides Are Not Always Bad

Strides are not automatically wrong.

They appear naturally in real tensor programs:

```text
slicing
transposes
channel layouts
batched data
padding
views without copies
```

The point is not:

```text
never use strides
```

The point is:

```text
know when a stride changes the memory access pattern
```

Sometimes the right fix is to change the kernel.

Sometimes it is to make the tensor contiguous before the kernel.

Sometimes the strided layout is worth it because it avoids a larger copy.

GPU work often involves that tradeoff.

## A PyTorch-Like Example

A tensor can have the same values but different layout behavior.

Contiguous matrix:

```text
shape   = [height, width]
strides = [width, 1]
```

Transposed view:

```text
shape   = [width, height]
strides = [1, width]
```

The transposed view may not copy the data.

It may only change how logical indices map to memory.

That means this logical access:

```text
x[row, col]
```

may no longer mean:

```cpp
row * width + col
```

It may mean:

```cpp
row * row_stride + col * col_stride
```

This is why Week 03 introduced strides before Week 06 introduced coalescing.

Strides describe the layout.

Coalescing describes whether nearby threads use that layout efficiently.

## The Mental Model

When reading a memory-heavy kernel, ask:

```text
Do neighboring threads read neighboring addresses?
Do neighboring threads write neighboring addresses?
Is the tensor contiguous or strided?
Does the fastest-changing thread index match the fastest-changing memory index?
Could this kernel be correct but slow because of layout?
```

For row-major data, the most common friendly pattern is:

```text
neighboring threads move across columns
```

Because columns are the fastest-changing dimension in row-major layout.

The real lesson of Week 06 is:

```text
Memory bandwidth is not only about how many bytes move.
It is also about how those bytes are arranged.
```

Week 05 taught the size of the traffic.

Week 06 teaches the shape of the traffic.
