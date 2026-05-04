# Week 10 Naive Reduction Kernels

Status: scaffolded

## Kernel Pattern

Record the one-thread-per-output reduction pattern.

## Row Sum

Write how a naive row-sum kernel chooses a row, loops across columns, and writes
`out[row]`.

## Row Max

Write how row max differs from row sum, especially the initial accumulator
value.

## Limitation

Explain why this kernel is correct but not fully parallel inside each row.
