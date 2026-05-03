# 7-Day GPU Kernels From Scratch Challenge

## Course Promise

In 7 days, you will build your first GPU kernels project from scratch.

This is for ML engineers who use PyTorch but want to understand what is happening underneath: threads, memory, indexing, correctness tests, benchmarks, and simple CUDA kernels.

You do not need to become a CUDA expert in 7 days.

You need to build one small project that proves you understand the basics.

By the end, you will have:

- one working CUDA `vector_add` kernel
- one extra elementwise kernel like ReLU, square, scale, or multiply
- correctness tests against a CPU, NumPy, or PyTorch reference
- one benchmark table
- one short explanation of grids, blocks, threads, indexing, and bounds checks
- one Skool submission for feedback

## How To Use This Challenge

Each day has four parts:

- Lesson: read this first
- Code: copy or adapt the snippet
- Tasks: do the work
- Ship: post or save the artifact

Minimum rule:

Every day, produce one small visible thing. Do not disappear into theory.

If you get stuck, submit the broken version and explain where it fails. That is a valid Skool post.

## Setup

Use the project repo if you already have it:

```bash
python -m pip install -e ".[dev]"
pytest
python examples/reference_bench.py
```

If you are writing a standalone CUDA file, you can compile it like this:

```bash
nvcc vector_add.cu -O2 -o vector_add
./vector_add
```

If you do not have a CUDA GPU yet, still do Day 1, Day 2, and the CPU/PyTorch reference parts. You can submit the mental model and reference tests first.

## Day 1: GPU Mental Model

### Lesson

A CPU is good at running a smaller number of complex instruction streams.

A GPU is good at running many simple operations in parallel.

When you write a GPU kernel, you write the code for one thread. Then the GPU runs many copies of that thread across your data.

For vector add:

```text
C[i] = A[i] + B[i]
```

each thread can handle one index `i`.

That is why vector add is the perfect first kernel. The math is boring, so you can focus on the GPU execution model.

### Mental Model

If `N = 1000`, you need to compute 1000 output values.

Instead of one CPU loop doing:

```text
for i in range(1000):
    C[i] = A[i] + B[i]
```

the GPU can launch many threads where each thread asks:

```text
Which index am I responsible for?
```

Then it computes one output element.

### Code

CPU reference:

```cpp
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

### Tasks

1. Run the repo tests or run a simple CPU reference.
2. Write a 5-sentence answer to this question: why does vector add fit the GPU?
3. Write one sentence explaining what a kernel is.
4. Write one sentence explaining what one GPU thread should do in vector add.
5. Record whether you have CUDA available locally.

### Questions

- What does one GPU thread compute?
- Why is vector add parallel?
- What part still feels mysterious: memory, launch syntax, indexing, or benchmarking?

### Ship

Post or save:

```text
Day 1:
My current GPU mental model:

One GPU thread does:

CUDA available? yes/no:

What confused me:
```

## Day 2: Grids, Blocks, Threads, And Indexing

### Lesson

CUDA launches threads in groups.

The basic hierarchy is:

```text
grid -> blocks -> threads
```

A block is a group of threads.

A grid is a group of blocks.

Inside a 1D kernel, each thread can calculate its global index like this:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Meaning:

- `threadIdx.x` is the thread's position inside its block
- `blockIdx.x` is the block's position inside the grid
- `blockDim.x` is how many threads are in each block
- `blockIdx.x * blockDim.x` skips over the previous blocks
- adding `threadIdx.x` gives the final global index

### Example

If:

```text
blockDim.x = 256
blockIdx.x = 0
threadIdx.x = 5
```

then:

```text
i = 0 * 256 + 5 = 5
```

If:

```text
blockDim.x = 256
blockIdx.x = 3
threadIdx.x = 5
```

then:

```text
i = 3 * 256 + 5 = 773
```

### Bounds Check

If `N = 1000` and `block_size = 256`, you need:

```text
ceil(1000 / 256) = 4 blocks
```

That launches:

```text
4 * 256 = 1024 threads
```

But you only have 1000 elements.

So the last 24 threads must do nothing.

That is why kernels usually need:

```cpp
if (i < n) {
    c[i] = a[i] + b[i];
}
```

### Tasks

1. Calculate how many blocks you need for `N = 1000` and `block_size = 256`.
2. Calculate how many extra threads get launched.
3. Write the indexing formula from memory.
4. Explain why the bounds check exists.
5. Pick another input size and repeat the calculation.

### Questions

- What does `blockIdx.x` mean?
- What does `threadIdx.x` mean?
- Why do we multiply by `blockDim.x`?
- What breaks if the bounds check is missing?

### Ship

Post or save:

```text
Day 2:
Indexing formula:

For N = 1000 and block size = 256:
- blocks launched:
- threads launched:
- extra threads:

Why the bounds check is needed:
```

## Day 3: First CUDA Kernel

### Lesson

Today you implement the first real kernel.

A CUDA kernel is marked with:

```cpp
__global__
```

That means it runs on the GPU and is launched from CPU code.

The kernel itself should describe what one thread does.

### Kernel Code

```cpp
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

### Launch Code

```cpp
int block_size = 256;
int num_blocks = (n + block_size - 1) / block_size;

vector_add_kernel<<<num_blocks, block_size>>>(d_a, d_b, d_c, n);
cudaDeviceSynchronize();
```

The formula:

```cpp
(n + block_size - 1) / block_size
```

is integer ceiling division. It gives enough blocks to cover all elements.

### Minimal Full File Shape

Use this structure for `vector_add.cu`:

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000;
    size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    vector_add_kernel<<<num_blocks, block_size>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    std::cout << "c[0] = " << h_c[0] << std::endl;
    std::cout << "c[999] = " << h_c[999] << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

Compile:

```bash
nvcc vector_add.cu -O2 -o vector_add
./vector_add
```

Expected:

```text
c[0] = 3
c[999] = 3
```

### Tasks

1. Create or update `vector_add.cu`.
2. Implement the kernel.
3. Allocate device memory.
4. Copy host arrays to device.
5. Launch the kernel.
6. Copy output back to host.
7. Print at least the first and last values.

### Questions

- Which arrays live on the CPU?
- Which arrays live on the GPU?
- What does `cudaMemcpyHostToDevice` do?
- What does `cudaMemcpyDeviceToHost` do?

### Ship

Post or save:

```text
Day 3:
Kernel compiles? yes/no:
First output value:
Last output value:
Bug I hit:
```

## Day 4: Correctness Testing

### Lesson

Printing one value is not enough.

A kernel is only useful if you can prove it is correct across different sizes.

Your test should compare GPU output against a trusted reference.

The reference can be CPU C++, NumPy, or PyTorch.

### CPU Check

Simple C++ check:

```cpp
bool check_result(const std::vector<float>& a,
                  const std::vector<float>& b,
                  const std::vector<float>& c) {
    for (int i = 0; i < static_cast<int>(c.size()); i++) {
        float expected = a[i] + b[i];
        if (std::abs(c[i] - expected) > 1e-5f) {
            std::cout << "Mismatch at " << i
                      << ": got " << c[i]
                      << ", expected " << expected << std::endl;
            return false;
        }
    }
    return true;
}
```

Add:

```cpp
#include <cmath>
```

Then after copying `h_c` back:

```cpp
bool ok = check_result(h_a, h_b, h_c);
std::cout << "correct: " << (ok ? "yes" : "no") << std::endl;
```

### Test Multiple Sizes

Change your program so it can test multiple sizes:

```cpp
std::vector<int> sizes = {1, 17, 256, 1000, 1000000};

for (int n : sizes) {
    // allocate, run kernel, copy back, check result
}
```

The important tests are odd sizes like:

```text
17
1000
```

because they reveal missing bounds checks.

### Tasks

1. Add a correctness checker.
2. Test `N = 1`.
3. Test `N = 17`.
4. Test `N = 256`.
5. Test `N = 1000`.
6. Test `N = 1_000_000`.
7. Record pass/fail for each size.

### Questions

- Which size is most likely to catch a bounds-check bug?
- Why is `N = 256` not enough?
- What should your test print when a mismatch happens?

### Ship

Post or save:

```text
Day 4 correctness:

N = 1:
N = 17:
N = 256:
N = 1000:
N = 1_000_000:

Did any size fail?
What did I change to fix it?
```

## Day 5: Benchmarking

### Lesson

Benchmarking GPU code is easy to do badly.

Common mistakes:

- timing only one run
- timing without synchronization
- comparing tiny GPU workloads to CPU loops
- saying "GPU is faster" without shape, dtype, and device

For this challenge, keep the benchmark simple.

Use `cudaEvent_t` to time the GPU kernel.

### GPU Timing Snippet

```cpp
cudaEvent_t start;
cudaEvent_t stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
vector_add_kernel<<<num_blocks, block_size>>>(d_a, d_b, d_c, n);
cudaEventRecord(stop);

cudaEventSynchronize(stop);

float milliseconds = 0.0f;
cudaEventElapsedTime(&milliseconds, start, stop);

std::cout << "GPU kernel time ms: " << milliseconds << std::endl;

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### CPU Timing Snippet

```cpp
#include <chrono>

auto cpu_start = std::chrono::high_resolution_clock::now();
vector_add_cpu(h_a.data(), h_b.data(), h_ref.data(), n);
auto cpu_end = std::chrono::high_resolution_clock::now();

double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
std::cout << "CPU time ms: " << cpu_ms << std::endl;
```

### Warmup

Run the GPU kernel a few times before timing:

```cpp
for (int warmup = 0; warmup < 5; warmup++) {
    vector_add_kernel<<<num_blocks, block_size>>>(d_a, d_b, d_c, n);
}
cudaDeviceSynchronize();
```

### Benchmark Table

Use a table like this:

```text
Device:
Dtype: float32
Block size: 256
Timing method: cudaEvent for GPU, chrono for CPU

| N | CPU ms | GPU kernel ms | Correct? |
| --- | ---: | ---: | --- |
| 1,000 | | | |
| 1,000,000 | | | |
| 10,000,000 | | | |
```

### Tasks

1. Add GPU timing with CUDA events.
2. Add CPU timing with `std::chrono`.
3. Benchmark at least three input sizes.
4. Include one small size and one large size.
5. Write down your device name if you know it.

### Questions

- When does the GPU start to look useful?
- Why might a tiny input be slower on GPU?
- Are you timing only the kernel or also memory copies?
- What would make this benchmark more honest?

### Ship

Post or save:

```text
Day 5 benchmark:

Device:
Dtype:
Block size:
Timing method:

| N | CPU ms | GPU ms | Correct? |
| --- | ---: | ---: | --- |
| | | | |
| | | | |
| | | | |

One thing I learned:
```

## Day 6: Second Elementwise Kernel

### Lesson

Most simple elementwise kernels share the same structure.

Only the math changes.

Vector add:

```cpp
c[i] = a[i] + b[i];
```

Square:

```cpp
out[i] = x[i] * x[i];
```

ReLU:

```cpp
out[i] = x[i] > 0.0f ? x[i] : 0.0f;
```

Scale:

```cpp
out[i] = alpha * x[i];
```

The indexing and bounds check stay the same.

### ReLU Kernel

```cpp
__global__ void relu_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}
```

### Square Kernel

```cpp
__global__ void square_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = x[i] * x[i];
    }
}
```

### Tasks

1. Pick one extra kernel: ReLU, square, scale, or multiply.
2. Implement it using the same indexing formula.
3. Add a CPU reference.
4. Test odd sizes.
5. Benchmark it once.
6. Explain what changed compared to `vector_add`.

### Questions

- Which parts of the kernel stayed the same?
- Which line changed?
- Is this kernel memory-bound or compute-heavy?
- What other PyTorch operation could you implement this way?

### Ship

Post or save:

```text
Day 6:
Second kernel:
Correctness result:
Benchmark result:
What stayed the same as vector_add:
What changed:
```

## Day 7: Package And Submit In Skool

### Lesson

The final skill is packaging.

A good project is not just code. It is code plus proof.

Your proof should include:

- what you built
- how you tested it
- what you measured
- what you learned
- what you still do not understand

That is what makes the project reviewable.

### Submission Template

Post this in Skool:

```text
Challenge: 7-Day GPU Kernels From Scratch

What I built:
- vector_add:
- second kernel:

Correctness:
- sizes tested:
- all passed? yes/no:

Benchmark:
- device:
- dtype:
- block size:
- timing method:

| N | CPU ms | GPU ms | Correct? |
| --- | ---: | ---: | --- |
| | | | |
| | | | |
| | | | |

My explanation of indexing:

Why the bounds check is needed:

What confused me:

What I want reviewed:
```

### Tasks

1. Clean up your code enough that someone else can read it.
2. Add a short README or notes section.
3. Fill in the submission template.
4. Post it in Skool.
5. Offer to review one other member's submission.

### Questions

- What is the most important thing you learned?
- What bug taught you the most?
- What would you improve next?
- Can you explain your kernel without reading the code?

### Ship

Your final Skool submission.

## Minimum Version

If you are busy, finish only this:

- `vector_add` works
- one odd-size test passes
- one benchmark exists
- you explain this formula:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

That is enough to participate.

## Stretch Version

If you want more:

- compare block sizes like 128, 256, and 512
- add a PyTorch baseline
- implement both ReLU and square
- plot input size vs runtime
- time memory copies separately from kernel time
- rewrite `vector_add` in Triton

## What We Will Check In Skool

When you submit, we will check:

- does the kernel compute the right result?
- does it handle odd input sizes?
- does it have a correct bounds check?
- does the benchmark include shape, dtype, device, and timing method?
- can you explain the indexing formula?
- did you ask a specific question?

Bad question:

```text
I do not understand CUDA.
```

Better question:

```text
My kernel works for N = 256 but fails for N = 1000. I think my bounds check or block count is wrong. Can someone check my indexing?
```

## Closing

This challenge is not about becoming elite in 7 days.

It is about getting your first real GPU artifact shipped:

- code
- correctness
- benchmark
- explanation

Start simple.

Make it correct.

Post it for review.
