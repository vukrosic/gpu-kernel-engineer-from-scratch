# Week 02 Vector Add

## What Was Built

Describe the vector-add note you wrote and how it connects the reference
implementation to the GPU kernel shape. Mention the one-worker-one-element
mapping.

## Correctness Check

Record the input you used for the scratch run and the reference output you
expected. Note whether you compared against NumPy or against
`gputriton.reference.vector_add`.

## Benchmark Or Observation

Add the shape or size you tried and any observation about how the work scales.
If you wrote a CUDA or Triton sketch, note what part was still only
pseudocode.

## Lesson Learned

Explain why vector add is a good first kernel and what it teaches about memory
reads and writes.

## Limitation Or Next Step

Write the next thing you would change if you were extending the kernel:
indexing, bounds, or a bigger input size.

## Write Down

- Why is vector add the simplest useful GPU kernel?
- Why do we compare against a CPU/NumPy reference first?
- What would a GPU version change about the memory access pattern?
- What would you test next if the output looked wrong?
