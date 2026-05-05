# Week 17: Softmax Math For Kernels

Softmax turns scores into probabilities.

For one row of scores:

```text
[2.0, 1.0, 0.1]
```

Softmax produces positive values that sum to 1:

```text
[0.659, 0.242, 0.099]
```

Week 17 teaches the math in the exact order a kernel will need it.

The point is not only to know the formula.

The point is to see softmax as:

```text
row max reduction
exp
row sum reduction
divide
```

## The Basic Formula

For scores `x`, softmax is:

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

For each element:

```text
take exp of this score
divide by the sum of all exp scores in the row
```

Python version:

```python
import math

def softmax_basic(xs):
    exps = [math.exp(x) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]
```

This is mathematically fine.

It is not numerically safe.

## The Overflow Problem

Exponential grows very fast.

This is dangerous:

```python
math.exp(1000.0)
```

A computer cannot represent that as a normal finite float.

So a naive softmax can overflow.

The input scores do not need to be absurd in real models. During training or
bad initialization, logits can become large enough to cause trouble.

GPU kernels should not rely on friendly inputs.

## The Max-Shift Trick

Stable softmax subtracts the row maximum first:

```text
shifted_i = x_i - max(x)
```

Then:

```text
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

This does not change the final probabilities.

It only makes the exponentials safer.

After subtracting the max:

```text
largest shifted value = 0
```

So the largest exponential is:

```text
exp(0) = 1
```

That prevents overflow from the largest value.

## Stable Python Version

The safe reference implementation is:

```python
import math

def softmax_stable(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]
```

Read it in kernel order:

```text
1. find row max
2. subtract row max
3. exponentiate
4. sum exponentials
5. divide each exponential by the sum
```

Those are the same steps a GPU kernel must organize.

## Why The Max-Shift Does Not Change The Answer

Subtracting the same value from every score multiplies every exponential by the
same constant.

```text
exp(x_i - m) = exp(x_i) / exp(m)
```

The numerator is divided by `exp(m)`.

The denominator is also divided by `exp(m)`.

The common factor cancels.

So:

```text
exp(x_i - m) / sum(exp(x_j - m))
```

gives the same probabilities as:

```text
exp(x_i) / sum(exp(x_j))
```

but with safer intermediate values.

## A Hand Example

Scores:

```text
[2, 1, 0]
```

Max:

```text
2
```

Shifted:

```text
[0, -1, -2]
```

Exponentials:

```text
[1.000, 0.368, 0.135]
```

Sum:

```text
1.503
```

Normalize:

```text
[1.000 / 1.503, 0.368 / 1.503, 0.135 / 1.503]
```

Output:

```text
[0.665, 0.245, 0.090]
```

The values are positive.

They sum to 1.

The largest score got the largest probability.

## Softmax Is Row-Wise In ML Kernels

In deep learning, softmax is usually applied across one row at a time.

For a matrix:

```text
[
  [2, 1, 0],
  [0, 3, 1],
]
```

Softmax is computed separately for each row:

```text
row 0 has its own max and sum
row 1 has its own max and sum
```

That means a softmax kernel is usually shaped like:

```text
one row or row tile per block
```

Inside that row, the kernel needs reductions.

## Softmax As Kernel Stages

A row-wise softmax kernel has this structure:

```text
load row values
reduce to row max
compute exp(x - row_max)
reduce to row sum
divide each exp by row_sum
write output row
```

Two stages are reductions:

```text
row max
row sum
```

Two stages are elementwise:

```text
exp
divide
```

That is why softmax is a bridge between earlier lessons.

It uses:

```text
indexing
memory layout
reductions
warp/block cooperation
numerical stability
```

## A CPU Reference With Rows

Reference code for a 2D row-wise softmax:

```python
import math

def row_softmax_reference(x):
    out = []

    for row in x:
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        total = sum(exps)
        out.append([v / total for v in exps])

    return out
```

This is the function a kernel can be tested against.

The reference should be simple and trusted.

The GPU kernel can be optimized later.

## Masked Values

Attention often uses masks.

A mask can say:

```text
this position should not receive probability
```

The usual math trick is to set masked scores to a very negative value before
softmax:

```text
masked score -> negative infinity
```

Then:

```text
exp(negative infinity) = 0
```

So the masked position contributes nothing to the sum and receives probability
zero.

This will matter more in attention lessons.

For now, remember:

```text
softmax often includes masking in real transformer kernels
```

## What Can Be Reused

A naive implementation might write intermediate exponentials to global memory:

```text
read scores
write exps
read exps
write probabilities
```

A better kernel tries to keep row data closer:

```text
registers or shared memory hold row values
reductions happen inside the block or warp
outputs are written once
```

This is why Week 17 comes after reductions.

Softmax performance depends heavily on how row data moves.

## The Core Pattern

When reading or writing softmax code, ask:

```text
What is one softmax row?
How is the row max computed?
Where are shifted exponentials stored?
How is the row sum computed?
When is each output divided by the sum?
Are masked positions handled safely?
Is the implementation numerically stable?
```

Softmax is simple as math and subtle as a kernel.

The stable math is the foundation.

## Bridge To Week 18

Week 18 turns this math into fused softmax thinking.

The central question becomes:

```text
how much work can one kernel do while avoiding unnecessary global memory traffic?
```
