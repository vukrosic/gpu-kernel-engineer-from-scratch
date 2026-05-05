# Week 41: Attention Forward Pass

Week 39 taught attention scores and masks.

Week 40 placed attention inside transformer dataflow.

Week 41 connects the full forward pass:

```text
scores -> masked softmax -> weighted sum with V
```

## Step 1: Start With The Formula

Single-head attention is:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

The tensors are:

```text
Q: [seq_q, head_dim]
K: [seq_k, head_dim]
V: [seq_k, head_dim]
output: [seq_q, head_dim]
```

The output has one vector for each query position.

## Step 2: Compute Scores

Scores compare every query with every key:

```python
scores = q @ k.transpose(-1, -2)
```

The shape is:

```text
[seq_q, head_dim] @ [head_dim, seq_k] -> [seq_q, seq_k]
```

Then scale:

```python
scores = scores / math.sqrt(head_dim)
```

This is the part Week 39 isolated.

## Step 3: Apply The Mask

For causal attention, future keys are hidden:

```python
rows = torch.arange(seq_q, device=q.device)[:, None]
cols = torch.arange(seq_k, device=q.device)[None, :]
scores = scores.masked_fill(cols > rows, -float("inf"))
```

After this, invalid positions are still present in the tensor, but their values
make softmax assign them zero probability.

Masking belongs before softmax.

## Step 4: Apply Softmax

Softmax converts scores into probabilities:

```python
probs = torch.softmax(scores, dim=-1)
```

Each query row becomes a distribution over key positions:

```text
probs[q, :] sums to 1
```

Numerically stable softmax is usually computed as:

```text
subtract row max
exponentiate
sum exponentials
divide by row sum
```

That connects attention back to the softmax lessons.

## Step 5: Multiply By V

The output is a weighted sum of value vectors:

```python
out = probs @ v
```

Shape:

```text
[seq_q, seq_k] @ [seq_k, head_dim] -> [seq_q, head_dim]
```

For one query position:

```text
out[q, :] = sum over k of probs[q, k] * V[k, :]
```

This is why softmax probabilities matter: they decide how much each value vector
contributes.

## Step 6: Write The Reference

A readable PyTorch reference is:

```python
def attention_forward(q, k, v, *, causal=False):
    head_dim = q.shape[-1]
    scores = q @ k.transpose(-1, -2)
    scores = scores / math.sqrt(head_dim)

    if causal:
        seq_q, seq_k = scores.shape[-2:]
        rows = torch.arange(seq_q, device=q.device)[:, None]
        cols = torch.arange(seq_k, device=q.device)[None, :]
        scores = scores.masked_fill(cols > rows, -float("inf"))

    probs = torch.softmax(scores, dim=-1)
    return probs @ v
```

This reference materializes `scores` and `probs`.

That is fine for clarity.

Optimized attention kernels try to avoid materializing those full matrices.

## Step 7: Think About Kernel Ownership

A naive attention implementation can be viewed as three kernels:

```text
kernel 1: compute scores
kernel 2: softmax rows
kernel 3: multiply probabilities by V
```

That is easy to understand but memory-heavy.

The large intermediates are:

```text
scores: [seq_q, seq_k]
probs:  [seq_q, seq_k]
```

For long sequences, these matrices dominate memory traffic.

That problem leads directly to FlashAttention concepts.

## The Core Pattern

Attention forward is:

```text
compute QK^T scores
scale scores
apply masks
softmax each score row
multiply probabilities by V
return one output vector per query
```

The simple reference teaches the math.

The optimized kernel changes how much intermediate data is written to memory.

## Bridge To Week 42

Week 42 teaches the FlashAttention idea.

Instead of materializing the full score and probability matrices, the kernel
processes attention in tiles and keeps the softmax state online.
