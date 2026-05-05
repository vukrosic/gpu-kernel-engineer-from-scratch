# Week 37: GELU Fusion

Week 37 starts transformer-adjacent kernels.

The first idea is fusion.

Fusion means combining operations that would otherwise be separate GPU kernels.

This week uses bias plus GELU as the example.

## Step 1: See The Unfused Path

In PyTorch, an MLP block often does something like:

```python
x = linear_output + bias
y = torch.nn.functional.gelu(x)
```

That is clear code.

But as GPU work, it may mean:

```text
kernel 1: add bias
kernel 2: apply GELU
```

The intermediate tensor `x` is written to memory and then read again.

Fusion tries to avoid that extra memory traffic.

## Step 2: Understand GELU

GELU is an activation function.

A common approximation is:

```python
def gelu_approx(x):
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))
```

You do not need to memorize the constants.

The important point is:

```text
GELU reads one value and returns one value
```

That makes it an elementwise operation.

Elementwise operations are natural candidates for fusion.

## Step 3: Fuse Bias And Activation

The fused operation is:

```text
output = gelu(input + bias)
```

As scalar logic:

```python
def fused_bias_gelu_scalar(x, bias):
    v = x + bias
    c = 0.7978845608028654
    return 0.5 * v * (1.0 + math.tanh(c * (v + 0.044715 * v * v * v)))
```

The kernel should load `x`, load `bias`, add them, apply GELU, and store the
final output.

It should not store the intermediate `x + bias` tensor.

## Step 4: Track The Memory Traffic

The unfused path has this shape:

```text
read input
read bias
write intermediate
read intermediate
write output
```

The fused path has this shape:

```text
read input
read bias
write output
```

The math is the same.

The memory traffic is different.

That is the main reason fusion matters.

## Step 5: Handle Bias Broadcasting

Bias is often shaped like the hidden dimension:

```text
input: [batch, hidden]
bias:  [hidden]
```

Each row uses the same bias vector.

For a flattened tensor, the hidden index is:

```python
hidden_index = offset % hidden_size
```

Then the bias load is:

```python
bias_value = bias[hidden_index]
```

This is the key indexing idea for fused bias operations.

The input position chooses both:

```text
which activation value to read
which bias value to reuse
```

## Step 6: Keep The Reference Simple

The PyTorch baseline should be direct:

```python
def torch_bias_gelu(x, bias):
    return torch.nn.functional.gelu(x + bias, approximate="tanh")
```

This is the correctness target.

The custom kernel can use different launch mechanics, but it should match this
operation.

## Step 7: Know What Fusion Can And Cannot Fix

Fusion helps most when memory traffic or launch overhead matters.

It may not help if:

```text
the tensor is tiny
the fused math is much more expensive
the operation is already fused by the framework
the custom kernel has poor memory access
```

Fusion is not magic.

It is a way to avoid unnecessary reads, writes, and launches when the operations
naturally belong together.

## The Core Pattern

For fused bias plus GELU:

```text
start from the PyTorch expression
identify the intermediate tensor
load input and bias once
compute bias add in registers
apply GELU immediately
store only the final output
compare to PyTorch with tolerance
benchmark against the unfused PyTorch path
```

Fusion is successful when it preserves the exact operation while reducing the
amount of intermediate memory traffic.

## Bridge To Week 38

Week 38 continues fusion, but with residual and normalization patterns.

Those appear constantly in transformer blocks, and they teach a more complex
kind of fusion than a single activation.
