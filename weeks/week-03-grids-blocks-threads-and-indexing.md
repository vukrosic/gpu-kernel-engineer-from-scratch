# Week 03: Grids, Blocks, Threads, And Indexing

Week 03 is about one idea:

```text
Every GPU thread needs to know which piece of data it owns.
```

The math can be simple. The indexing cannot be vague.

In Week 02, vector add looked like this:

```text
out[i] = a[i] + b[i]
```

This week explains how a GPU thread gets its `i`.

## The GPU Work Model

A CUDA kernel launches many threads at once.

Those threads are grouped like this:

```text
grid
  block 0
    thread 0
    thread 1
    thread 2
    ...
  block 1
    thread 0
    thread 1
    thread 2
    ...
```

Use this plain-language version:

```text
thread = one worker
block  = a group of workers
grid   = all workers launched for one kernel
```

A kernel does not automatically know which array element each thread should
handle. You write that mapping yourself.

That mapping is indexing.

## One Thread, One Output Element

The first useful pattern is:

```text
one thread computes one output element
```

For vector add:

```text
thread 0 computes out[0]
thread 1 computes out[1]
thread 2 computes out[2]
thread 3 computes out[3]
```

Each thread performs the same instruction, but on a different index:

```cpp
out[i] = a[i] + b[i];
```

The important question is:

```text
How does the thread compute i?
```

## The 1D Index Formula

CUDA gives each thread a few built-in values.

For now, focus on these three:

```cpp
blockIdx.x
threadIdx.x
blockDim.x
```

They mean:

```text
blockIdx.x   = which block am I in?
threadIdx.x  = which thread am I inside this block?
blockDim.x   = how many threads are in each block?
```

The standard 1D index formula is:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Read it as:

```text
skip the threads in previous blocks,
then add my position inside this block
```

Example with `blockDim.x = 4`:

```text
block 0, thread 0 -> i = 0 * 4 + 0 = 0
block 0, thread 1 -> i = 0 * 4 + 1 = 1
block 0, thread 2 -> i = 0 * 4 + 2 = 2
block 0, thread 3 -> i = 0 * 4 + 3 = 3

block 1, thread 0 -> i = 1 * 4 + 0 = 4
block 1, thread 1 -> i = 1 * 4 + 1 = 5
block 1, thread 2 -> i = 1 * 4 + 2 = 6
block 1, thread 3 -> i = 1 * 4 + 3 = 7
```

So blocks and threads become one continuous sequence of element indices.

## A Kernel-Shaped Version

This is the core of a 1D elementwise CUDA kernel:

```cpp
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
```

Line by line:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

This gives the current thread a global element index.

```cpp
if (i < n) {
```

This prevents extra threads from touching memory outside the array.

```cpp
out[i] = a[i] + b[i];
```

This is the actual work for one element.

Most beginner elementwise kernels have this shape:

```text
compute index
check bounds
read input
write output
```

## Why The Bounds Check Matters

Suppose the input has `10` elements and each block has `4` threads.

You cannot launch `2.5` blocks, so you launch `3` blocks:

```text
3 blocks * 4 threads = 12 threads
```

But valid indices are only:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

Threads with index `10` and `11` exist, but they do not own real data.

That is why this guard is not optional:

```cpp
if (i < n) {
    out[i] = a[i] + b[i];
}
```

Without it, the kernel may read or write beyond the end of the array. That is a
correctness bug, not just a performance issue.

## How Many Blocks To Launch

The CPU launches the kernel and chooses the grid size.

Common launch setup:

```cpp
int threads_per_block = 256;
int blocks = (n + threads_per_block - 1) / threads_per_block;
```

The formula rounds up.

For `n = 1000`:

```text
blocks = (1000 + 256 - 1) / 256
blocks = 1255 / 256
blocks = 4
```

Four blocks create:

```text
4 * 256 = 1024 threads
```

The extra `24` threads are harmless because of the bounds check.

Then the launch looks like:

```cpp
add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_out, n);
```

Read the triple angle brackets as:

```text
launch this kernel with this many blocks and this many threads per block
```

## The Same Pattern Works For Many Operations

Once each thread has an index, the math can change while the structure stays
the same.

Vector add:

```cpp
out[i] = a[i] + b[i];
```

Multiply:

```cpp
out[i] = a[i] * b[i];
```

Square:

```cpp
out[i] = x[i] * x[i];
```

ReLU:

```cpp
out[i] = x[i] > 0.0f ? x[i] : 0.0f;
```

These are all elementwise kernels.

They differ in the operation, but they share the same indexing pattern:

```text
one thread owns one output position
```

## From 1D Vectors To 2D Matrices

GPU memory is linear. Even a 2D matrix is stored as a 1D sequence.

For a matrix with `3` rows and `4` columns:

```text
[
  [ 0,  1,  2,  3],
  [ 4,  5,  6,  7],
  [ 8,  9, 10, 11],
]
```

Row-major layout stores the first row, then the second row, then the third row:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
```

To convert `(row, col)` into a flat index:

```cpp
int index = row * width + col;
```

For `row = 2`, `col = 1`, and `width = 4`:

```text
index = 2 * 4 + 1 = 9
```

So this matrix position:

```text
matrix[2][1]
```

maps to this flat memory position:

```text
matrix[9]
```

## 2D Indexing In A Kernel

For a 2D problem, you can give each thread a row and a column.

The formulas look like this:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

Then convert the row and column to one flat index:

```cpp
int i = row * width + col;
```

A kernel-shaped version:

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

The structure is still simple:

```text
compute row and column
check bounds
convert to flat index
read input
write output
```

## 1D Launch Versus 2D Launch

A 1D launch is enough for a flat vector:

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
```

A 2D launch is easier to read for matrix-shaped work:

```cpp
dim3 threads(16, 16);
dim3 blocks(
    (width + threads.x - 1) / threads.x,
    (height + threads.y - 1) / threads.y
);
```

That creates blocks with:

```text
16 * 16 = 256 threads per block
```

Each thread gets one `(row, col)` position.

The `x` direction usually maps to columns. The `y` direction usually maps to
rows.

## Indexing Bugs To Recognize

Most early CUDA bugs are indexing bugs.

Common mistakes:

```text
forgetting the bounds check
using blockIdx.x when you meant threadIdx.x
using height where you meant width
writing row + col instead of row * width + col
launching too few blocks
having two threads write the same output index
```

The dangerous part is that a kernel can compile and still be wrong.

That is why reference implementations matter. A CPU or NumPy reference gives you
the expected output, so the GPU result has something honest to match.

## The Core Mental Model

Do not think of a GPU kernel as one function call doing one thing.

Think of it as many workers running the same function at the same time.

Each worker asks:

```text
Where am I in the grid?
Which data element do I own?
Am I inside the valid input range?
What output should I write?
```

For elementwise kernels, the clean answer is:

```text
one valid thread writes one valid output element
```

Once that is clear, vector add, multiply, square, ReLU, and many preprocessing
operations become variations of the same idea.
