# Week 40: Transformer Kernel Dataflow

Week 40 is not a checkpoint.

It is the lesson that connects the transformer operations from Weeks 37-39 into
one dataflow picture.

GPU engineering is easier when you can see where tensors are created, reused,
and written back to memory.

## Step 1: Start With A Transformer Block

A simplified transformer block looks like:

```text
x
attention path
residual add
normalization
MLP path
residual add
normalization
```

In code, that often becomes:

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

Different models arrange normalization differently, but the same building
blocks keep appearing:

```text
matmul
softmax
elementwise activation
normalization
residual add
```

## Step 2: Separate Compute-Heavy And Memory-Heavy Work

Matmul is usually compute-heavy when it is well tiled:

```text
Q projection
K projection
V projection
MLP up projection
MLP down projection
attention output projection
```

Elementwise and row-wise operations are often memory-heavy:

```text
bias add
GELU
residual add
RMSNorm or LayerNorm
dropout
masking
```

That distinction guides optimization.

Compute-heavy work needs good tiling and hardware utilization.

Memory-heavy work often benefits from fusion.

## Step 3: Look For Intermediate Tensors

Every intermediate tensor has a cost.

For an MLP:

```text
hidden = x @ W1
hidden = hidden + bias
hidden = gelu(hidden)
out = hidden @ W2
```

The obvious intermediate is:

```text
hidden after bias but before GELU
```

Week 37 fused that pattern.

For residual norm:

```text
tmp = x + residual
out = norm(tmp)
```

Week 38 studied that temporary value.

## Step 4: Track Attention Dataflow

Attention creates large logical intermediates:

```text
scores = QK^T
probs = softmax(scores)
out = probs V
```

For long sequences, `scores` and `probs` can be huge.

That is why attention kernels care so much about tiling and memory reuse.

The key question is:

```text
do we need to materialize the full score matrix?
```

Naive attention says yes.

FlashAttention-style kernels try to avoid it.

## Step 5: Match Kernel Type To Operation Type

Use the operation shape to choose the kernel strategy:

```text
elementwise op      -> one output element per position
row-wise reduction  -> one program or block per row
matmul              -> one program or block per output tile
attention           -> tiled score/probability/value pipeline
```

This map helps you avoid designing every kernel from scratch.

Most transformer kernels are combinations of patterns you have already learned.

## Step 6: Know Where Fusion Helps

Fusion is useful when operations are close together and reuse the same data.

Good fusion candidates:

```text
bias + activation
residual + norm
scale + mask + softmax
dropout + residual
```

Fusion is less useful when it makes the kernel too large, too register-heavy, or
hard to test.

The practical question is:

```text
does fusion reduce memory traffic without making the kernel fragile?
```

## Step 7: Keep The Baseline In The Picture

Transformer kernels can become hard to reason about.

Always keep a clear PyTorch expression nearby:

```python
scores = q @ k.transpose(-1, -2)
scores = scores * scale
scores = scores.masked_fill(mask, -float("inf"))
probs = torch.softmax(scores, dim=-1)
out = probs @ v
```

This reference is the contract.

The optimized kernel changes data movement, not the meaning.

## The Core Pattern

When looking at transformer performance:

```text
draw the tensor dataflow
mark compute-heavy matmuls
mark memory-heavy elementwise and row-wise operations
find intermediate tensors
ask which intermediates can be fused away
keep the PyTorch reference as the correctness contract
profile one suspected bottleneck at a time
```

This is the bridge from individual kernels to real ML systems.

## Bridge To Week 41

Week 41 implements the full attention forward path conceptually:

```text
scores
mask
softmax
weighted sum with V
```

You already know the pieces. The next lesson connects them in order.
