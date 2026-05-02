# Week 17

Status: writing template

## What To Capture

- one tiny score vector
- the max shift you subtracted
- the probabilities and their sum

## Hand-Worked Example

| Score | Shifted Score | Exp | Probability |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

## What Was Built

Describe the softmax example you wrote or studied. Name the input scores and
whether you used the stable max-shift version.

## Correctness Check

Record why the probabilities sum to 1. If you compared stable and unstable
softmax, write down what changed.

## Benchmark Or Observation

If you measured anything, note whether the example exposed underflow or
overflow behavior. If you did not measure, write the comparison you would make.

## Lesson Learned

Summarize softmax in plain language.

## Limitation Or Next Step

Write one sentence about why math alone is not the whole implementation story.

## Write-Back Prompts

1. What does softmax turn scores into?
2. Why do we subtract the max?
3. Where does numerical stability matter most?
