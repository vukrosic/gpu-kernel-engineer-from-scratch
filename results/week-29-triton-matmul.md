# Week 29

Status: writing prompt

## What Was Built

- Describe the Triton matmul sketch or note you wrote.
- Name the tile shape and the one output region it owns.
- Mention whether you borrowed the shape directly from the CUDA version.

## Correctness Check

- Explain why the tile accumulation matches the reference math.
- State the input shapes or tile sizes you used to think through correctness.
- Note one place where a mask or boundary condition matters.

## Benchmark Or Observation

- Record the command you ran and the shape you used.
- Note what you were trying to learn from the observation.
- If you did not benchmark, write the exact comparison you would run next.

## Lesson Learned

- Finish this sentence: "Matmul becomes easier to think about when ..."
- Capture one thing Triton expresses more directly than CUDA.

## Limitation Or Next Step

- Name the tile choice you still want to compare.
- Write the next tuning question Week 30 should answer.
