# Week 39: Attention Scores And Masks

Week 39 starts attention from the smallest useful pieces.

The full attention formula is:

```text
softmax(QK^T / sqrt(d)) V
```

This lesson focuses on the first half:

```text
QK^T / sqrt(d)
masking
```

## Step 1: Name The Tensors

For one attention head:

```text
Q: [seq_q, head_dim]
K: [seq_k, head_dim]
V: [seq_k, head_dim]
```

`Q` contains query vectors.

`K` contains key vectors.

`V` contains value vectors.

The score matrix compares every query position with every key position:

```text
scores: [seq_q, seq_k]
```

## Step 2: Compute One Score

One score is a dot product:

```text
score[q, k] = dot(Q[q, :], K[k, :])
```

As a loop:

```python
score = 0.0
for d in range(head_dim):
    score += Q[q, d] * K[k, d]
```

This is matmul logic.

The attention score matrix is `Q @ K.T`.

## Step 3: Apply The Scale

Attention scores are usually scaled:

```python
scores = scores / math.sqrt(head_dim)
```

The scale keeps dot products from growing too large as `head_dim` grows.

Large scores can make softmax too sharp and numerically unstable.

For kernels, the scale is just a multiply:

```python
scale = 1.0 / math.sqrt(head_dim)
score = score * scale
```

## Step 4: Add A Causal Mask

In autoregressive decoding, position `q` cannot attend to future positions.

That means:

```text
key position k is invalid when k > q
```

A causal mask turns those invalid scores into negative infinity:

```python
if k > q:
    score = -float("inf")
```

Softmax will turn negative infinity into zero probability.

That is why masks are applied before softmax.

## Step 5: Add A Padding Mask

Padding masks handle fake tokens added to make sequences the same length.

If a key position is padding, every query should ignore it:

```python
if key_is_padding[k]:
    score = -float("inf")
```

Causal masks and padding masks can both apply.

The final rule is:

```text
if a key position is not allowed, its score becomes negative infinity
```

## Step 6: Build The PyTorch Reference

A clear reference is:

```python
def attention_scores(q, k, *, causal=False):
    head_dim = q.shape[-1]
    scores = q @ k.transpose(-1, -2)
    scores = scores / math.sqrt(head_dim)

    if causal:
        seq_q, seq_k = scores.shape[-2:]
        rows = torch.arange(seq_q, device=scores.device)[:, None]
        cols = torch.arange(seq_k, device=scores.device)[None, :]
        scores = scores.masked_fill(cols > rows, -float("inf"))

    return scores
```

This reference is not optimized.

It exists to make the math and masking rules explicit.

## Step 7: Think Like A Kernel

A score kernel owns tiles of the score matrix.

One tile might cover:

```text
BLOCK_Q query positions
BLOCK_K key positions
```

For each tile, the kernel needs:

```text
Q block: [BLOCK_Q, head_dim]
K block: [BLOCK_K, head_dim]
output scores: [BLOCK_Q, BLOCK_K]
```

This is tiled matmul with attention-specific masking.

## The Core Pattern

Attention score computation is:

```text
load query vectors
load key vectors
compute dot products
scale by 1 / sqrt(head_dim)
apply causal and padding masks
produce score tile
```

Understanding this piece makes the full attention forward pass much less
mysterious.

## Bridge To Week 40

Week 40 zooms out to transformer kernel dataflow.

Before implementing full attention, it helps to see where attention, MLPs,
normalization, residuals, and fusion fit in the larger block.
