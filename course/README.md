# Course Home

This folder is the operating manual for the 12-month GPU Kernels From Scratch
course.

## Files

- [syllabus.md](syllabus.md): the full month-by-month and week-by-week course.
- [weekly-rhythm.md](weekly-rhythm.md): how each weekly flagship video maps to 7 days of work.
- [recovery-system.md](recovery-system.md): what to do when you miss a day, week, or month.

## Weekly Files

The actual course path is in `../weeks/`. Each week gets one file that you can
follow from top to bottom.

- [Week 01: GPU Mental Model And Baseline](../weeks/week-01-gpu-mental-model.md)

## Monthly Pages

- [Month 01: GPU Foundations](month-01-gpu-foundations.md)
- [Month 02: Memory And Benchmarking](month-02-memory-and-benchmarking.md)
- [Month 03: Reductions](month-03-reductions.md)
- [Month 04: Scans, Atomics, And Synchronization](month-04-scans-atomics-synchronization.md)
- [Month 05: Softmax And Normalization](month-05-softmax-and-normalization.md)
- [Month 06: Matmul Foundations](month-06-matmul-foundations.md)
- [Month 07: Triton For AI Kernels](month-07-triton-for-ai-kernels.md)
- [Month 08: Triton Matmul And Tuning](month-08-triton-matmul-and-tuning.md)
- [Month 09: PyTorch Integration](month-09-pytorch-integration.md)
- [Month 10: Transformer Kernels](month-10-transformer-kernels.md)
- [Month 11: Attention And Inference](month-11-attention-and-inference.md)
- [Month 12: Portfolio And Interviews](month-12-portfolio-and-interviews.md)

## Design Principles

- One flagship lesson per week is the backbone.
- Each lesson ships one portfolio artifact.
- Every fourth week is a checkpoint week.
- Assignments have Minimum, Standard, and Stretch versions.
- Transformers are used as context for kernels, not as a competing parallel course.

## Definition Of Done

A week is done when you can show:

- the core concept in your own words
- a correct implementation or reference experiment
- a test against a trusted baseline
- a benchmark or measured observation
- a short portfolio note explaining what changed and why it matters
