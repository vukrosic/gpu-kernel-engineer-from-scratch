# Week 45: Benchmark Dashboard

Week 45 follows the [Week 44 attention capstone plan](week-44-attention-capstone-plan.md)
and teaches how to present benchmark evidence.

The goal is not to make a pretty table.

The goal is to make performance claims easy to verify.

## Step 1: Decide What A Row Means

Each dashboard row should describe one comparison.

The row is not just a number.

It is a small claim:

```text
for this operation, shape, dtype, hardware, and timing method, this result was
observed against this baseline
```

That sentence tells you which columns the table needs.

## Step 2: Use Stable Columns

A useful benchmark dashboard can start with:

```text
operation
implementation
baseline
shape
dtype
hardware
timing method
correctness status
result
note
```

These columns protect you from vague claims.

They force every number to carry context.

## Step 3: Separate Correctness From Speed

A fast incorrect kernel is not a win.

The dashboard should show correctness before performance:

```text
correctness: pass
timing: 32 us
```

If correctness is unknown, the row should say so:

```text
correctness: not checked
timing: do not trust yet
```

That keeps the dashboard honest.

## Step 4: Record Shape Precisely

Shape is part of the result.

For vector add:

```text
n = 1,048,576
```

For matmul:

```text
M = 1024, K = 1024, N = 1024
```

For attention:

```text
batch = 1, heads = 8, seq = 2048, head_dim = 64
```

Without shape, a benchmark number is almost useless.

## Step 5: Name The Baseline

Always say what the custom path is compared against:

```text
PyTorch torch.add
PyTorch matmul
unfused PyTorch bias + GELU
naive attention reference
previous tile size
```

Different baselines answer different questions.

Comparing to PyTorch asks:

```text
is the custom path competitive with the normal ML workflow?
```

Comparing to an earlier kernel asks:

```text
did this optimization improve my own implementation?
```

## Step 6: Track Timing Method

Timing method belongs in the dashboard.

Examples:

```text
CUDA events, 100 repeats
torch.utils.benchmark
Nsight Systems timeline
manual wall-clock timing
```

Some methods are better than others, but hidden timing methods are always bad.

If CUDA work is involved, synchronization matters.

The dashboard should make that visible.

## Step 7: Keep Notes Short

The note column should explain the main interpretation:

```text
edge shape handled by mask
larger tile improved reuse but used more registers
KV cache saves projection work but increases memory
```

Do not turn the dashboard into a diary.

Longer reasoning belongs in the weekly result notes.

The dashboard is the index.

## Example Dashboard Row

```text
operation: bias + GELU
implementation: fused Triton sketch
baseline: PyTorch x + bias then GELU
shape: batch=32, hidden=4096
dtype: float32
hardware: local GPU
timing method: CUDA events, 50 repeats
correctness: pass
result: pending
note: fusion removes one intermediate activation write
```

Even with `result: pending`, this row is useful because the comparison is clear.

## The Core Pattern

A good benchmark dashboard records:

```text
what was measured
what it was compared against
which shape and dtype were used
which timing method produced the number
whether correctness passed
what one-line interpretation matters
```

Benchmarks should make the project easier to trust.

If a table makes the project look faster but harder to verify, it is doing the
wrong job.

## Bridge To Week 46

Week 46 turns the benchmark and lesson work into interview explanations.

The next skill is explaining what you built clearly enough that another
engineer can follow the decisions.
