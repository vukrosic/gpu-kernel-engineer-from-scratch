# Week 43: KV Cache

Week 43 shifts from full-sequence attention to inference.

During generation, the model produces one token at a time.

The KV cache stores previous keys and values so the model can reuse them.

## Step 1: Compare Training And Decoding

During training or prefill, the model often sees a full sequence:

```text
tokens: [t0, t1, t2, ..., tS]
```

Attention computes many query positions at once.

During decoding, the model receives one new token:

```text
new token: tS+1
```

Only the new query is needed.

But it must attend to all previous keys and values.

## Step 2: See What Would Be Wasteful

Without a cache, each decoding step would recompute keys and values for the
whole prefix:

```text
step 1: compute K/V for t0
step 2: compute K/V for t0, t1
step 3: compute K/V for t0, t1, t2
```

That repeats work.

The KV cache avoids this by storing:

```text
K for previous tokens
V for previous tokens
```

At each new step, the model computes only the new token's K and V, then appends
them to the cache.

## Step 3: Understand Cache Shape

A common cache layout is:

```text
K cache: [batch, heads, max_seq, head_dim]
V cache: [batch, heads, max_seq, head_dim]
```

For each generated token, one position is written:

```text
cache[:, :, current_position, :] = new_k_or_v
```

Then attention reads:

```text
all cached K/V positions from 0 to current_position
```

The write is small.

The read grows with sequence length.

## Step 4: Decode One Token

For one new token, attention looks like:

```text
q_new: [batch, heads, 1, head_dim]
k_cache: [batch, heads, seq_so_far, head_dim]
v_cache: [batch, heads, seq_so_far, head_dim]
```

The score shape is:

```text
[batch, heads, 1, seq_so_far]
```

The output shape is:

```text
[batch, heads, 1, head_dim]
```

Only one query position is active, but it attends across the whole cached
prefix.

## Step 5: Write The Simple Reference

A readable reference for one decode step is:

```python
def decode_attention_step(q, k_cache, v_cache, seq_len):
    k = k_cache[:, :, :seq_len, :]
    v = v_cache[:, :, :seq_len, :]
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
```

This shows the main cost:

```text
read cached K
read cached V
compute attention for one query
```

The longer the context, the more cache data must be read.

## Step 6: Know Why Layout Matters

The cache layout affects memory access.

Two common questions are:

```text
are head_dim values contiguous?
are sequence positions easy to stream through?
```

For decoding, the kernel repeatedly reads keys and values across the sequence
dimension.

If memory layout makes those reads awkward, attention becomes slower.

KV-cache performance is often a memory bandwidth and layout problem, not just a
math problem.

## Step 7: Track Cache Updates Separately

A decode step has two jobs:

```text
append new K/V to cache
read cache to compute attention output
```

It is useful to think about them separately.

The append path cares about writing the new token into the right cache position.

The attention path cares about reading all valid previous positions efficiently.

Mixing those two ideas too early makes debugging harder.

## The Core Pattern

KV cache inference is:

```text
compute new K and V
write them at the current cache position
read cached K and V up to the current position
compute attention for the new query
return one output vector for the new token
advance the cache position
```

The cache saves repeated projection work.

The new bottleneck is reading a growing history efficiently.

## Bridge To Week 44

Week 44 should package the attention and inference lessons into a capstone
direction.

At this point, the important story is clear: attention performance is about
math, memory traffic, tiling, and reuse all at once.
