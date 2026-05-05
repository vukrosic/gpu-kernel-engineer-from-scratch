# Week 44: Attention Capstone Plan

Week 41 taught attention forward.

Week 42 taught why FlashAttention avoids materializing huge score and
probability matrices.

Week 43 taught why KV cache matters during decoding.

Week 44 turns those ideas into one capstone story.

## Step 1: Choose The Capstone Question

A good capstone is not "I studied attention."

A better capstone asks one clear engineering question:

```text
how does attention performance change when memory traffic, tiling, and cached
K/V reuse become the main constraints?
```

That question connects the month:

```text
attention forward -> full math path
FlashAttention -> reduce intermediate memory
KV cache -> reuse past keys and values during inference
```

The capstone should explain the relationship between those three ideas.

## Step 2: Define The Baseline Path

The baseline path is the readable PyTorch version:

```python
def attention_reference(q, k, v, *, causal=False):
    d = q.shape[-1]
    scores = q @ k.transpose(-1, -2)
    scores = scores / math.sqrt(d)

    if causal:
        seq_q, seq_k = scores.shape[-2:]
        rows = torch.arange(seq_q, device=q.device)[:, None]
        cols = torch.arange(seq_k, device=q.device)[None, :]
        scores = scores.masked_fill(cols > rows, -float("inf"))

    probs = torch.softmax(scores, dim=-1)
    return probs @ v
```

This path is not the fastest.

It is the contract.

Every optimized explanation should point back to it.

## Step 3: Name The Memory Problem

Naive attention materializes large intermediates:

```text
scores: [seq_q, seq_k]
probs:  [seq_q, seq_k]
```

For long sequences, these tensors are the story.

The capstone should make the memory problem visible before discussing speed.

For example:

```text
if seq = 4096, scores has 16,777,216 elements per head
```

That one line explains why attention is not just a math formula.

It is also a memory system problem.

## Step 4: Explain The Tiled Alternative

The FlashAttention-style idea is:

```text
process attention in tiles
keep softmax state online
accumulate output directly
avoid writing full scores and probabilities
```

The capstone does not need to implement production FlashAttention to be useful.

It must explain what changes:

```text
naive path writes large intermediate matrices
tiled path keeps smaller blocks close to the computation
```

That is the engineering lesson.

## Step 5: Explain The Inference Alternative

During decoding, only one new query position is active.

The model still needs previous keys and values.

The KV cache changes the work from:

```text
recompute old K/V every step
```

to:

```text
append new K/V once
read cached K/V for attention
```

The capstone should be honest about the tradeoff:

```text
KV cache saves repeated computation
KV cache uses memory that grows with sequence length
```

That tradeoff is the inference story.

## Step 6: Pick The Evidence

The capstone should include evidence, not only explanations.

Useful evidence can be:

```text
shape table
memory estimate
correctness comparison
timing table
profiling observation
diagram of dataflow
```

Do not include every possible artifact.

Pick the evidence that answers the capstone question.

For this project, the cleanest evidence is usually:

```text
PyTorch reference
shape and memory estimate
kernel or concept note
benchmark or profiling note
```

## Step 7: Write The Capstone Flow

The capstone should read in this order:

```text
1. attention forward is the baseline math
2. naive attention materializes large intermediates
3. tiling reduces intermediate memory traffic
4. decoding reuses past K/V through a cache
5. the remaining bottleneck is reading long cached histories efficiently
```

That flow turns Weeks 41-43 into one engineering story.

## The Core Pattern

An attention capstone should answer:

```text
what is the reference operation?
which tensors become large?
which intermediate writes can be avoided?
which data can be reused during decoding?
what tradeoff does the optimized path introduce?
what evidence supports the explanation?
```

This is how a set of lessons becomes a project someone else can understand.

## Bridge To Week 45

Week 45 turns project evidence into a benchmark dashboard.

The dashboard is where correctness notes, shapes, baselines, and measurements
become easy to scan.
