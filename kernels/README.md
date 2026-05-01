# Kernel Topics

This folder organizes kernels by concept. Implementations can live in `cuda/`,
`triton_kernels/`, or `gputriton/` depending on the stage of the course.

## Core Kernels

- vector add
- elementwise add, multiply, square, ReLU
- copy, scale, axpy
- row sum and row max
- histogram or counting kernel
- prefix sum / scan
- softmax
- layer norm
- RMSNorm
- matmul
- fused bias and activation
- attention forward
- KV-cache simulation

## Kernel Card Template

For each kernel, document:

- purpose
- input and output shapes
- baseline implementation
- correctness strategy
- benchmark command
- known bottleneck
- next optimization
