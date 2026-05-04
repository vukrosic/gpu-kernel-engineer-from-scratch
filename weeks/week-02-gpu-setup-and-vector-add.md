# Week 02: GPU Setup And Vector Add

## What This Week Is

Week 02 turns the Week 01 mental model into the first kernel-shaped workflow.
The operation is still tiny:

```text
out[i] = a[i] + b[i]
```

That small formula is enough to teach the whole loop you will repeat for the
rest of the course:

1. define the CPU/NumPy reference
2. understand the GPU work mapping
3. run a correctness check
4. inspect or compile the CUDA version
5. record the result in `results/`

You do not need to become fluent in CUDA this week. You need to understand the
shape of one simple kernel from top to bottom.

## Lesson: From Array Formula To GPU Kernel

Read this section first. Everything you need for the Week 02 tasks is here.

### Vector Add As A Reference

Vector add takes two arrays of the same shape and creates a third array:

```text
a = [1, 2, 3]
b = [4, 5, 6]
out = [5, 7, 9]
```

Each output position depends on exactly one position from `a` and one position
from `b`:

```text
out[0] = a[0] + b[0]
out[1] = a[1] + b[1]
out[2] = a[2] + b[2]
```

The CPU/NumPy reference is short because NumPy already knows how to apply the
operation element by element:

```python
def vector_add(a, b):
    return a + b
```

The reference is the trusted answer. The GPU version is only correct if it
matches this result.

### Why Vector Add Fits A GPU

Vector add is a good first GPU kernel because the work is independent.

To compute `out[100]`, you do not need `out[99]`. To compute `out[100]`, you
only need:

```text
a[100]
b[100]
```

That means many workers can safely work at the same time:

```text
worker 0 computes out[0]
worker 1 computes out[1]
worker 2 computes out[2]
...
worker 100 computes out[100]
```

This is the first GPU pattern:

```text
one worker handles one output element
```

In CUDA language, those workers are threads.

### Threads, Blocks, And Grids

CUDA launches many threads. Threads are grouped into blocks. Blocks are grouped
into a grid.

For Week 02, use this plain-language model:

```text
thread: one worker
block: a group of workers
grid: all workers launched for this kernel
```

If you launch 4 blocks with 256 threads per block, you asked CUDA for:

```text
4 * 256 = 1024 threads
```

Each thread needs to know which element it owns. CUDA gives each thread two
important built-in values:

```text
blockIdx.x   which block am I in?
threadIdx.x  which thread am I inside this block?
```

The common 1D index formula is:

```text
idx = blockIdx.x * blockDim.x + threadIdx.x
```

Read that as:

```text
skip all threads in earlier blocks, then add my position inside this block
```

Example with `blockDim.x = 4`:

```text
block 0, thread 0 -> idx = 0 * 4 + 0 = 0
block 0, thread 1 -> idx = 0 * 4 + 1 = 1
block 0, thread 2 -> idx = 0 * 4 + 2 = 2
block 0, thread 3 -> idx = 0 * 4 + 3 = 3

block 1, thread 0 -> idx = 1 * 4 + 0 = 4
block 1, thread 1 -> idx = 1 * 4 + 1 = 5
```

That formula is the bridge between "many GPU threads exist" and "each output
element is written once."

### Why The Bounds Check Exists

The input length is often not a perfect multiple of the block size.

If `n = 10` and the block size is 4, then this grid size is common:

```text
grid_size = (n + block_size - 1) / block_size
```

Using integer division, that gives:

```text
grid_size = (10 + 4 - 1) / 4 = 13 / 4 = 3
```

Three blocks with four threads each creates 12 threads. But the array only has
10 valid positions: `0` through `9`.

So threads with `idx = 10` and `idx = 11` must do nothing. That is why the
kernel uses:

```cpp
if (idx < n) {
    out[idx] = a[idx] + b[idx];
}
```

Without the bounds check, extra threads could read or write outside the array.
That is a correctness bug.

### Host Memory And Device Memory

The CPU and GPU have separate memory spaces in this CUDA example.

The host arrays live on the CPU side:

```cpp
std::vector<float> h_a(n), h_b(n), h_out(n);
```

The device arrays live on the GPU side:

```cpp
float* d_a = nullptr;
float* d_b = nullptr;
float* d_out = nullptr;
```

The naming convention is important:

- `h_` means host
- `d_` means device

The CUDA vector-add program follows this data path:

```text
1. create host inputs
2. allocate device arrays
3. copy host inputs to device
4. launch the GPU kernel
5. wait for the GPU to finish
6. copy device output back to host
7. compare host output against expected answer
```

That movement matters. A GPU kernel is not just the math line. It also includes
the setup, memory allocation, copies, launch, synchronization, and validation
around the math line.

### The Kernel Line By Line

The provided CUDA starter contains this kernel:

```cpp
__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
```

Read it as:

- `__global__` means this function runs on the GPU and is launched by the CPU
- `a` and `b` are input arrays in GPU memory
- `out` is the output array in GPU memory
- `n` is the number of valid elements
- `idx` chooses which element this thread owns
- `if (idx < n)` protects the array bounds
- `out[idx] = a[idx] + b[idx]` performs one element of work

This is the smallest useful CUDA kernel shape. Later kernels get more complex,
but this pattern keeps coming back: compute an index, guard the bounds, read
inputs, write output.

### Setup Checks

There are two valid paths this week.

Path A: CPU-only machine

You can still complete the week by running the Python reference and reading the
CUDA file. Your result note should say that CUDA compilation was not available
on your machine.

Path B: CUDA-capable machine

You can compile and run the CUDA starter with `nvcc`. `nvcc` is NVIDIA's CUDA
compiler. If `nvcc --version` works, you can try the compile command in this
week's command section.

Do not hide setup problems. Record them. A setup note is still a useful result
because GPU systems work includes environment reality.

## What You Need From The Repo

- [../gputriton/reference.py](../gputriton/reference.py)
- [../gputriton/bench.py](../gputriton/bench.py)
- [../tests/test_reference.py](../tests/test_reference.py)
- [../course/month-01-gpu-foundations.md](../course/month-01-gpu-foundations.md)
- [../cuda/vector_add.cu](../cuda/vector_add.cu)
- [../triton_kernels/vector_add.py](../triton_kernels/vector_add.py)

Inspect them in this order:

1. `gputriton/reference.py` to see the trusted vector-add answer.
2. `tests/test_reference.py` to see how the reference is checked.
3. `cuda/vector_add.cu` to trace the host/device CUDA workflow.
4. `gputriton/bench.py` to see where the reference benchmark size comes from.
5. `triton_kernels/vector_add.py` only as a preview of the later Triton path.

While reading `cuda/vector_add.cu`, find these lines:

- where host arrays are created
- where device arrays are allocated
- where host data is copied to the device
- where `block_size` and `grid_size` are chosen
- where the kernel is launched
- where the result is copied back
- where correctness is checked

## Exact Commands

Run the Python reference path first:

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
print("expected:", np.array([5.0, 7.0, 9.0], dtype=np.float64))
PY
```

If you have CUDA and `nvcc`, compile and run the CUDA starter:

```bash
mkdir -p build
nvcc -O2 -std=c++14 cuda/vector_add.cu -o build/vector_add
./build/vector_add
./build/vector_add 10
./build/vector_add 1048576
```

If `nvcc` is not available, write that down in the result note and continue with
the file-reading and Python-reference path.

## Build This

Create or update `results/week-02-vector-add.md` so it becomes a real Week 02
artifact.

It should include:

- the Python reference output
- whether CUDA compilation was available
- if CUDA was available, the compile command and program output
- a short explanation of the one-thread-one-element mapping
- a short explanation of the bounds check
- one sentence about what could go wrong if the output was incorrect

Use this outline:

```markdown
# Week 02 Vector Add

## Environment

- Machine:
- Python:
- CUDA available:
- `nvcc --version` result, if available:

## Python Reference

Input:

Output:

What this proves:

## CUDA Starter

Compile command:

Run output:

If CUDA was unavailable, explain what blocked it.

## Kernel Mapping

Explain `idx = blockIdx.x * blockDim.x + threadIdx.x`.

## Bounds Check

Explain why `if (idx < n)` is needed.

## What I Would Debug First

If the result were wrong, what would you inspect first?
```

## Code Sketch

The CPU version is:

```python
def vector_add(a, b):
    out = []
    for x, y in zip(a, b):
        out.append(x + y)
    return out
```

The GPU-shaped version is:

```text
for each GPU thread:
    idx = global thread index
    if idx is inside the array:
        out[idx] = a[idx] + b[idx]
```

The actual CUDA kernel expresses that GPU-shaped version as:

```cpp
__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
```

Do not rush past this code. This is the first real kernel pattern in the course.

## Write Down

Answer these in your note:

1. Why is vector add the simplest useful GPU kernel?
2. Why do we compare against a CPU/NumPy reference first?
3. What does the output shape tell you about the work being done?
4. What does `idx = blockIdx.x * blockDim.x + threadIdx.x` compute?
5. Why does the kernel need `if (idx < n)`?
6. What changes when the same logic runs on the GPU instead of in a CPU loop?
7. If the output looked wrong, what would you check first?

Use this file to answer the concepts. Use the repo files to answer what the
project actually does.

## Minimum

- `pytest` runs, or you record why setup blocked it
- `python examples/reference_bench.py` runs, or you record why setup blocked it
- `results/week-02-vector-add.md` exists
- you explain vector add in your own words
- you explain one-thread-one-element mapping in your own words

## Standard

- you run the tiny Python vector-add experiment
- you inspect `cuda/vector_add.cu` and identify the host/device steps
- you explain the global index formula
- you explain the bounds check
- you record whether CUDA compilation was available

## Stretch

- you compile and run `cuda/vector_add.cu` on a CUDA machine
- you run the CUDA program with at least two input sizes
- you compare the vector-add benchmark to the matmul benchmark from Week 01
- you explain why "easy math" does not automatically mean "easy GPU performance"

## If You Are Behind

Do the Python reference run, inspect the CUDA kernel, and write the result note.
Do not skip the explanation of `idx`. That formula is the main Week 02 idea.

## Done Checklist

Minimum:

- [ ] `results/week-02-vector-add.md` exists
- [ ] It includes the Python reference output
- [ ] It explains vector add in plain language
- [ ] It explains one-thread-one-element mapping

Standard:

- [ ] It identifies the host arrays and device arrays in `cuda/vector_add.cu`
- [ ] It explains the global index formula
- [ ] It explains the bounds check
- [ ] It records CUDA availability

Stretch:

- [ ] You compiled `cuda/vector_add.cu` with `nvcc`
- [ ] You ran the CUDA program with two input sizes
- [ ] You wrote one debugging hypothesis for a wrong output

## Next Week

Week 03 keeps the same indexing idea and expands it into tensor shapes,
row-major layout, strides, and flat memory addresses. Week 02 is the seed:
compute the global index, guard the bounds, read inputs, write output.
