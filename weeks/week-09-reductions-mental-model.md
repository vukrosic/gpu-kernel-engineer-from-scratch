# Week 09: Reductions Mental Model

Weeks 04 through 08 were mostly about elementwise kernels and memory behavior.

Elementwise kernels usually have this shape:

```text
one input position -> one output position
```

Week 09 starts a new kind of problem:

```text
many input positions -> fewer output positions
```

That is a reduction.

## What A Reduction Does

A reduction combines multiple values into a smaller result.

Sum is a reduction:

```text
[1, 2, 3, 4] -> 10
```

Max is a reduction:

```text
[1, 7, 3, 4] -> 7
```

Min is a reduction:

```text
[5, 2, 9, 4] -> 2
```

Mean is also a reduction:

```text
[2, 4, 6, 8] -> 5
```

The input has many values.

The output has fewer values.

That shape is the main difference from elementwise work.

## Elementwise Vs Reduction

Elementwise add:

```text
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]

out = [6, 8, 10, 12]
```

Each output position has a matching input position:

```text
out[0] = a[0] + b[0]
out[1] = a[1] + b[1]
out[2] = a[2] + b[2]
out[3] = a[3] + b[3]
```

Reduction sum:

```text
x = [1, 2, 3, 4]

out = 10
```

The output depends on all input positions:

```text
out = x[0] + x[1] + x[2] + x[3]
```

That is why reductions are more complicated for GPUs.

Elementwise work can assign one thread to one output.

Reduction work needs multiple values to combine into one output.

## Reductions Shrink Shape

Shape is the easiest way to recognize a reduction.

For a vector:

```text
input shape:  [4]
output shape: []
```

The empty shape means a scalar.

For a matrix:

```text
input shape:  [2, 4]
```

Row sum over columns:

```text
output shape: [2]
```

Column sum over rows:

```text
output shape: [4]
```

A reduction removes or shrinks one dimension.

That is the first mental model:

```text
reduction changes the output shape
```

## Row Sum

Take this matrix:

```text
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
]
```

Row sum means sum each row:

```text
row 0: 1 + 2 + 3 + 4 = 10
row 1: 5 + 6 + 7 + 8 = 26
```

The output is:

```text
[10, 26]
```

Input shape:

```text
2 rows x 4 columns
```

Output shape:

```text
2 values
```

One row becomes one output value.

## Row Max

Use the same matrix:

```text
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
]
```

Row max means take the maximum value in each row:

```text
row 0: max(1, 2, 3, 4) = 4
row 1: max(5, 6, 7, 8) = 8
```

The output is:

```text
[4, 8]
```

Row sum and row max use different operations.

But their shape is the same:

```text
one row -> one output value
```

## Axis

Frameworks like NumPy and PyTorch describe reductions with an axis.

For a matrix with shape:

```text
[rows, columns]
```

Reducing over axis `1` means reducing across columns inside each row:

```text
row sum
```

The output keeps the row dimension:

```text
[rows]
```

Reducing over axis `0` means reducing down rows for each column:

```text
column sum
```

The output keeps the column dimension:

```text
[columns]
```

Axis tells you which dimension disappears.

That is the rule to remember:

```text
the reduced axis is the one that gets collapsed
```

## A Reduction Chain

The most direct way to reduce is a chain:

```text
total = 0
total = total + x[0]
total = total + x[1]
total = total + x[2]
total = total + x[3]
```

Python-shaped code:

```python
def sum_1d(x):
    total = 0.0
    for value in x:
        total += value
    return total
```

This is correct and easy to understand.

It is also mostly serial.

Each step depends on the previous value of `total`.

That dependency is what makes reductions different from elementwise kernels.

## A Reduction Tree

A reduction does not have to combine values one at a time.

It can combine pairs:

```text
[1, 2, 3, 4]
```

First layer:

```text
1 + 2 = 3
3 + 4 = 7
```

Second layer:

```text
3 + 7 = 10
```

Tree view:

```text
      10
     /  \
    3    7
   / \  / \
  1  2 3  4
```

The tree shape matters because it creates places where parallel workers can
cooperate.

That cooperation becomes the core GPU problem.

## Why Reductions Need Coordination

In an elementwise kernel, each thread can write its own output element:

```text
thread 0 -> out[0]
thread 1 -> out[1]
thread 2 -> out[2]
```

In a reduction, multiple input values contribute to the same output.

For row sum:

```text
x[0, 0]
x[0, 1]
x[0, 2]
x[0, 3]
```

All contribute to:

```text
out[0]
```

If multiple threads update `out[0]` carelessly, they can overwrite each other.

That is why reductions need a plan for coordination.

The plan can involve:

```text
one thread doing the whole reduction
multiple threads producing partial sums
shared memory
warp-level operations
atomics
multiple kernel launches
```

Week 09 only needs the mental model.

The later weeks teach the coordination tools.

## Reductions In Machine Learning

Reductions show up everywhere in ML systems.

Examples:

```text
sum of losses over a batch
max value in softmax
sum of exponentials in softmax
mean and variance in layer norm
row sums in attention masks
argmax for predictions
gradient accumulation
```

Many "AI kernels" are not purely elementwise.

They combine values across rows, columns, batches, channels, or sequence
positions.

That is why reductions are a major step in the course.

## Correctness Questions

Reduction correctness has a few extra questions.

For sum:

```text
what is the initial value?
what dtype is used for accumulation?
what order are values added in?
what tolerance is acceptable for floating-point comparison?
```

For max:

```text
what is the initial value?
what happens for negative values?
what happens for empty inputs?
what happens when values tie?
```

Elementwise kernels also need correctness checks, but reductions add shape and
accumulation details.

## The Mental Model

When reading a reduction, ask:

```text
What is the input shape?
Which axis is being reduced?
What is the output shape?
What operation combines the values?
What is the initial value?
Do multiple workers need to cooperate on one output?
```

For row sum:

```text
input shape:  [rows, columns]
axis:         columns
output shape: [rows]
operation:    addition
initial:      0
cooperation:  values in one row combine into one output
```

The real lesson of Week 09 is:

```text
reductions shrink data, and shrinking data requires coordination
```

Week 10 will use this mental model to read the first naive reduction kernels.
