# Week 46: Interview Explanations

Week 46 teaches how to explain GPU kernel work in an interview.

The point is not to memorize answers.

The point is to turn technical work into clear engineering reasoning.

## Step 1: Start With The Problem

A strong answer begins with the problem, not the tool.

Weak opening:

```text
I used Triton and CUDA.
```

Better opening:

```text
I worked on making custom GPU kernels correct, measurable, and easier to compare
against PyTorch baselines.
```

The second version tells the listener what the work was for.

Tools come after purpose.

## Step 2: Use The Reference-First Story

The course has one strong repeated pattern:

```text
reference first
custom kernel second
benchmark third
```

That is a good interview answer because it sounds like engineering discipline.

Example:

```text
For each kernel, I started with a PyTorch or CPU reference, used it as the
correctness contract, then compared shape, dtype, device, and values before
timing the custom path.
```

This answer explains how you avoided optimizing wrong code.

## Step 3: Explain Memory Traffic

Many kernel improvements are memory stories.

A concise explanation:

```text
Some operations are limited less by arithmetic and more by how many times they
read and write global memory. Fusion helps when it removes intermediate tensor
writes without changing the operation.
```

Then give one example:

```text
Bias plus GELU can be fused so the kernel reads input and bias, computes the
activation immediately, and writes only the final output.
```

That connects concept to implementation.

## Step 4: Explain Tiling

Tiling is about reuse.

A clear answer:

```text
In matmul, a tile of A and a tile of B can be loaded once and reused for many
multiply-adds. The tile size controls reuse, but it also affects shared memory,
register pressure, and occupancy.
```

This answer is good because it includes the tradeoff.

Do not say only "tiling is faster."

Say why and what it costs.

## Step 5: Explain Attention

Attention needs a memory-aware explanation.

A useful version:

```text
Naive attention materializes the full score and probability matrices. For long
sequences, those intermediates become large. FlashAttention-style kernels avoid
writing the full matrices by processing tiles and keeping online softmax state.
```

That answer covers:

```text
the baseline
the bottleneck
the optimization idea
```

## Step 6: Explain KV Cache

KV cache is an inference tradeoff.

A clean answer:

```text
During decoding, the model only gets one new token at a time. The KV cache saves
past keys and values so they do not need to be recomputed every step. It saves
compute, but it uses memory that grows with sequence length.
```

Again, include the tradeoff.

Interviewers listen for tradeoffs because real systems are not free wins.

## Step 7: Use A Three-Part Answer

For most questions, use this shape:

```text
1. what problem did I face?
2. what approach did I use?
3. how did I check it?
```

Example:

```text
The problem was that a fused operation can look faster while being wrong. I used
a PyTorch baseline as the contract, tested output shape, dtype, device, and
values, and only then compared timing. That kept performance work tied to
correctness.
```

That is short, specific, and believable.

## The Core Pattern

Strong GPU-kernel interview explanations include:

```text
the operation being optimized
the baseline used for correctness
the bottleneck hypothesis
the optimization idea
the tradeoff
the evidence
```

If you can say those six things simply, the project becomes much easier to
understand.

## Bridge To Week 47

Week 47 turns these explanations into a project story and resume bullets.

The same rule applies: specific claims, clear evidence, and no vague speedup
theater.
