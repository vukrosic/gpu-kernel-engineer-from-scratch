# Week 30: Triton Matmul Performance Knobs

Week 29 showed the basic Triton matmul tile.

Week 30 teaches the knobs that change its performance.

The main knobs are:

```text
BLOCK_M
BLOCK_N
BLOCK_K
num_warps
num_stages
```

These knobs change reuse, parallelism, register pressure, and scheduling.

## Block Sizes

`BLOCK_M` and `BLOCK_N` choose the output tile size:

```text
BLOCK_M x BLOCK_N values of C
```

Larger output tiles can reuse A and B values more.

They also create larger accumulator tiles.

For example:

```text
64 x 64 accumulator = 4096 values
128 x 64 accumulator = 8192 values
```

The compiler maps this work across lanes and registers, but the pressure still
matters.

## BLOCK_K

`BLOCK_K` controls how much of the reduction dimension each loop step covers.

Small `BLOCK_K`:

```text
more loop iterations
smaller A and B tiles
```

Large `BLOCK_K`:

```text
fewer loop iterations
larger loaded tiles
more pressure
```

There is no universal best value.

The best setting depends on matrix shapes and hardware.

## num_warps

`num_warps` controls how many warps are used by a Triton program.

More warps can help a larger tile expose enough parallel work.

Too many warps can waste resources or increase overhead.

Small tiles often need fewer warps.

Large tiles often need more.

The knob is tied to the shape of the program's work.

## num_stages

`num_stages` affects pipelining.

Matmul repeatedly loads tiles and computes dot products:

```text
load next tiles
compute current tiles
```

Pipelining tries to overlap those phases.

More stages can improve overlap.

More stages can also increase resource use.

Again, the setting must be measured.

## Autotuning

Autotuning means trying several configurations and choosing the best measured
one for a shape or family of shapes.

Example search space:

```python
configs = [
    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4},
    {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 8},
]
```

The point is not to try random settings forever.

The point is to define a reasonable search space and measure fairly.

## Shape Dependence

A config that wins for:

```text
M = 4096, N = 4096, K = 4096
```

may not win for:

```text
M = 128, N = 4096, K = 4096
```

Skinny matrices, small batches, and unusual K sizes can change the best choice.

That is why tuned kernels often specialize for shape families.

## What Fair Measurement Requires

Compare configs with:

```text
same input shapes
same dtype
same warmup
same repeat count
same correctness check
same timing method
```

Do not compare one lucky run against one unlucky run.

Use median or stable summaries.

Report the shape with the result.

Without the shape, a matmul benchmark is barely meaningful.

## The Core Pattern

When looking at a tuning result, ask:

```text
What shape was tested?
What configs were compared?
Which knob changed?
Did correctness pass for every config?
Was timing measured fairly?
Does the winner make sense for the tile shape?
```

Tuning is not magic.

It is disciplined comparison.

## Bridge To Week 31

Week 31 adds a batch dimension.

The next question is:

```text
how does indexing change when many matmuls are packed together?
```
