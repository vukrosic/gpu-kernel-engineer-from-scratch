# Week 03: Tensor Shapes, Memory Layout, And Indexing

Week 02 introduced the first 1D kernel pattern:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Week 03 moves past that first formula.

The new idea is:

```text
Tensors have shapes, but GPU memory is flat.
```

A matrix may look 2D in Python. A batch of images may look 4D in PyTorch. But
inside a CUDA kernel, the data usually arrives as a pointer to one long block of
memory.

Indexing is how you connect the logical tensor shape to the flat memory address.

## Shape Is Not Storage

This matrix has shape `3 x 4`:

```text
[
  [ 0,  1,  2,  3],
  [ 4,  5,  6,  7],
  [ 8,  9, 10, 11],
]
```

As a human, you see rows and columns.

In memory, the values are commonly stored as one flat sequence:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
```

The shape tells you how to interpret the sequence.

The memory itself is still linear.

## Row-Major Layout

Most C, C++, NumPy, and PyTorch tensors use row-major layout when they are
contiguous.

Row-major means:

```text
store row 0
then row 1
then row 2
...
```

For a matrix with `height = 3` and `width = 4`:

```text
row 0 -> flat positions 0, 1, 2, 3
row 1 -> flat positions 4, 5, 6, 7
row 2 -> flat positions 8, 9, 10, 11
```

The formula is:

```cpp
int index = row * width + col;
```

That formula says:

```text
skip all previous rows, then move to the column inside this row
```

Example:

```text
row = 2
col = 1
width = 4

index = row * width + col
index = 2 * 4 + 1
index = 9
```

So:

```text
matrix[2][1] maps to matrix[9]
```

## Why Width Matters

The width is the number of elements in one full row.

That is why this is correct:

```cpp
int index = row * width + col;
```

And this is wrong:

```cpp
int index = row + col;
```

For `(row = 2, col = 1)`, `row + col` gives:

```text
2 + 1 = 3
```

But position `3` is still in the first row.

The correct flat position is `9`.

This is the kind of bug that can make a kernel compile, run, and silently write
the wrong result.

## From Flat Index Back To Row And Column

Sometimes a kernel starts with one flat thread index and needs to recover the
2D position.

Given:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

You can recover row and column with:

```cpp
int row = i / width;
int col = i % width;
```

For `i = 9` and `width = 4`:

```text
row = 9 / 4 = 2
col = 9 % 4 = 1
```

So flat index `9` means:

```text
row 2, column 1
```

This pair of formulas is the inverse of row-major flattening:

```cpp
int i = row * width + col;
int row = i / width;
int col = i % width;
```

## Two Ways To Cover A Matrix

There are two common ways to map threads to a 2D matrix.

The first way is to use a flat 1D launch and convert `i` into `(row, col)`.

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < height * width) {
    int row = i / width;
    int col = i % width;
    out[i] = x[i];
}
```

This is simple and works well for many elementwise operations.

The second way is to use a 2D launch, where CUDA gives you an `x` direction and
a `y` direction.

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

Then you flatten the 2D position:

```cpp
int i = row * width + col;
```

The 2D version often reads more naturally for matrix-shaped work.

## A 2D ReLU Kernel

ReLU is a simple elementwise operation:

```text
out = max(x, 0)
```

For a matrix, each output element depends on one input element:

```text
out[row, col] = max(x[row, col], 0)
```

In CUDA-shaped code:

```cpp
__global__ void relu_2d_kernel(const float* x, float* out, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int i = row * width + col;
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}
```

The important part is not ReLU. The important part is the mapping:

```text
thread position -> row and column -> flat memory index
```

## Launch Shape For A 2D Kernel

A 2D kernel usually uses `dim3`.

```cpp
dim3 threads(16, 16);
```

This means each block has:

```text
16 threads in x
16 threads in y
16 * 16 = 256 total threads
```

The grid must cover the full matrix:

```cpp
dim3 blocks(
    (width + threads.x - 1) / threads.x,
    (height + threads.y - 1) / threads.y
);
```

The `x` dimension covers columns.

The `y` dimension covers rows.

For `width = 1000` and `threads.x = 16`:

```text
blocks.x = (1000 + 16 - 1) / 16 = 63
```

For `height = 600` and `threads.y = 16`:

```text
blocks.y = (600 + 16 - 1) / 16 = 38
```

The launch creates a little extra coverage at the edges. The bounds check
handles those extra threads:

```cpp
if (row < height && col < width) {
```

## Strides

So far, the lesson used contiguous row-major tensors.

A contiguous `height x width` matrix has this stride pattern:

```text
row stride = width
col stride = 1
```

That means:

```cpp
int index = row * width + col;
```

Can also be written as:

```cpp
int index = row * row_stride + col * col_stride;
```

For a contiguous matrix:

```cpp
int row_stride = width;
int col_stride = 1;
```

Strides tell you how far to move in flat memory when one logical index changes.

Move one column:

```text
index changes by 1
```

Move one row:

```text
index changes by width
```

This idea becomes important when tensors are sliced, transposed, or viewed
without copying data.

## Batch Dimensions

Deep learning tensors often have more than two dimensions.

For example, a batch of vectors might have shape:

```text
batch x features
```

If `batch = 3` and `features = 4`, the logical tensor looks like:

```text
sample 0: feature 0, feature 1, feature 2, feature 3
sample 1: feature 0, feature 1, feature 2, feature 3
sample 2: feature 0, feature 1, feature 2, feature 3
```

The flat index formula is the same as a matrix:

```cpp
int index = batch_id * features + feature_id;
```

For images, a common shape is:

```text
N x C x H x W
```

That means:

```text
N = batch
C = channels
H = height
W = width
```

For contiguous `NCHW` layout, the flat index is:

```cpp
int index = ((n * C + c) * H + h) * W + w;
```

Read it from left to right:

```text
choose the batch
then the channel
then the row
then the column
```

The formula looks bigger, but it is still the same idea:

```text
logical tensor position -> flat memory position
```

## Indexing Controls Correctness

In a GPU kernel, wrong indexing usually means wrong data.

Common mistakes:

```text
using height where width belongs
forgetting the channel dimension
using row + col instead of row * width + col
mixing up x and y dimensions
writing outside the valid shape
having multiple threads write the same output element
```

These bugs are dangerous because the code may still compile.

The kernel may even look fast.

But speed does not matter if the thread-to-data mapping is wrong.

## The Mental Model

When reading a kernel for a matrix or tensor, ask:

```text
What is the logical shape?
What logical element does this thread own?
How is that logical element converted into a flat index?
What bounds check protects the edge?
Does each output element get written exactly once?
```

For a 2D elementwise kernel, the clean answer is:

```text
one valid thread owns one valid row-column position
```

Then the flat index connects that position to memory:

```cpp
int i = row * width + col;
```

That is the real lesson of Week 03.

You are not just launching more threads. You are making a precise contract
between tensor shape, thread position, and memory address.
