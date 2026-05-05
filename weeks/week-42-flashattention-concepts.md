# Week 42: FlashAttention Concepts

Week 41 taught attention forward in the simple form:

```text
scores = QK^T
probs = softmax(scores)
out = probs V
```

That version is easy to understand.

It is also memory-hungry.

Week 42 teaches the core idea behind FlashAttention-style kernels.

## Step 1: See The Memory Problem

For sequence length `S`, the score matrix has shape:

```text
[S, S]
```

If `S = 4096`, then:

```text
scores has 4096 * 4096 = 16,777,216 elements
```

The probability matrix has the same size.

Naive attention may write and read both:

```text
write scores
read scores for softmax
write probs
read probs for probs @ V
```

FlashAttention tries to avoid materializing these full matrices in global
memory.

## Step 2: Process Attention In Tiles

Instead of computing all scores at once, process blocks of keys and values:

```text
load a block of Q rows
load a block of K rows
compute a score tile
update softmax state
load matching V rows
update output accumulator
move to the next K/V block
```

The output is built gradually.

The full score matrix never needs to be stored.

## Step 3: Remember Softmax Needs A Row Sum

Softmax for one row is:

```text
exp(score_i - max_score) / sum_j exp(score_j - max_score)
```

The row max and row sum are global across all keys.

If keys are processed in blocks, the kernel must update:

```text
running row max
running row sum
running output accumulator
```

That is the heart of online softmax.

## Step 4: Understand Online Softmax State

For each query row, keep:

```text
m: current maximum score seen so far
l: current sum of exponentials adjusted to m
o: current output accumulator
```

When a new score block arrives, compute its block max:

```text
m_new = max(m_old, block_max)
```

Then rescale the old state so it matches the new maximum:

```text
l_new = l_old * exp(m_old - m_new) + block_sum * exp(block_max - m_new)
```

The output accumulator is rescaled in the same spirit before adding the new
weighted values.

You do not need to derive the full kernel yet.

You need to see why the state exists.

## Step 5: Apply Masks Inside The Tile

Causal masks still apply.

But now they apply to a score tile:

```text
query positions: q block
key positions: k block
mask score[q, k] when k > q
```

Invalid scores become negative infinity before the tile participates in
softmax.

Masking is still before softmax.

Only the storage strategy changed.

## Step 6: Compare Naive And FlashAttention Dataflow

Naive attention:

```text
compute full scores
store full scores
softmax full rows
store full probabilities
multiply probabilities by V
```

FlashAttention-style attention:

```text
stream K/V blocks
compute score tiles
update online softmax state
accumulate output
store final output
```

The math is the same.

The memory traffic is different.

## Step 7: Know The Tradeoff

FlashAttention-style kernels are more complex because they combine:

```text
tiled matmul
masking
online softmax
V accumulation
careful numerical stability
```

The reward is avoiding huge intermediate matrices.

That matters most when sequence length is large.

For tiny sequences, the complexity may not pay off.

## The Core Pattern

FlashAttention is built around this idea:

```text
do not materialize scores and probabilities
process attention in tiles
keep online softmax state per query row
accumulate the final output directly
write only the output
```

This is the same lesson as earlier fusion weeks, but at attention scale:

```text
avoid unnecessary global-memory intermediates
```

## Bridge To Week 43

Week 43 moves from training-style attention to inference.

The next bottleneck is the KV cache: storing and reusing past keys and values so
the model does not recompute the entire prefix at every generated token.
