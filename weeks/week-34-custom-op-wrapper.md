# Week 34: Custom Op Wrapper

## What This Week Is

You sketch the boundary where a custom kernel meets PyTorch. The goal is to
understand the wrapper, not to hide the kernel behind magic, and to keep the
interface small enough that the kernel still feels visible.

## What To Read

- [../course/month-09-pytorch-integration.md](../course/month-09-pytorch-integration.md)
- [week-33-pytorch-baselines.md](week-33-pytorch-baselines.md)
- [../triton_kernels/vector_add.py](../triton_kernels/vector_add.py)
- [../triton_kernels/matmul.py](../triton_kernels/matmul.py)

## Exact Commands

```bash
pytest tests/test_reference.py tests/test_gpu_tracks.py
python examples/reference_bench.py
```

## Build This

Write `results/week-34-custom-op-wrapper.md` with the wrapper inputs, outputs,
and one note about what the wrapper should not do.

## Code Sketch

```python
def custom_op(x, y, *, backend="auto"):
    if backend == "reference":
        return reference_op(x, y)
    return kernel_op(x, y)
```

Write one sentence explaining why the sketch is correct before you optimize it.

## Write Down

- What lives inside the wrapper?
- What should stay in the kernel?
- How do you keep the interface simple?
- What stays identical across backends?

## Minimum

- one wrapper sketch
- one note file
- one plain-language summary

## Standard

- compare wrapper and kernel responsibilities
- note one validation step

## Stretch

- sketch an API doc stub
- explain one integration risk

## If You Are Behind

Keep the wrapper description short.

## Next Week

You will define a test matrix so the same idea can be checked in more than one way.
