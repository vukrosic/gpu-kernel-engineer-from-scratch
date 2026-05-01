# 50-Minute YouTube Roadmap Video Plan

## Video Goal

Create the anchor video for the entire course:

```text
The 12-Month Roadmap to Become a GPU Kernel Engineer
```

This video should make the viewer feel:

- GPU kernels are a valuable AI engineering skill.
- The path is hard but clear.
- They know exactly what to do for the next year.
- The GitHub repo is the free roadmap.
- Skool is the place to build it with other people.

The video is not a full CUDA lesson. It is the map.

## Best Title

```text
The 12-Month Roadmap to Become a GPU Kernel Engineer
```

## Backup Titles

- How I Would Learn GPU Kernels in 12 Months
- CUDA, Triton, AI Systems: The Complete GPU Kernel Roadmap
- From Python to GPU Kernels: A 1-Year AI Engineering Roadmap
- The GPU Kernel Engineering Roadmap for AI Engineers
- The AI Systems Skill Stack Nobody Teaches Clearly

## Thumbnail

Big text:

```text
GPU KERNEL ROADMAP
```

Visual:

```text
CUDA -> Triton -> Matmul -> Attention -> Portfolio
```

Small badge:

```text
12 MONTHS
```

Thumbnail concept:

- dark background
- five-step roadmap arrow
- GPU chip or server rack
- small code window with `matmul` / `softmax` / `attention`
- your face optional, pointing at the roadmap

## Pinned Comment

```text
Start here:
GitHub roadmap: https://github.com/vukrosic/gpu-kernel-engineer-from-scratch
Community: https://www.skool.com/become-ai-researcher-2669/about

Week 01 is already written as a follow-it-top-to-bottom plan. Start there, run
the repo, record your first benchmark, and do the minimum version if you are busy.
```

## Description Draft

```text
GPU kernel engineering is one of the most important skill stacks behind modern AI systems.

In this video, I walk through a complete 12-month roadmap for learning CUDA, Triton, GPU performance engineering, and AI kernels from scratch.

You will see exactly what to study each month, what to build each week, how to avoid getting overwhelmed, and how to turn the work into a public portfolio.

Free GitHub roadmap:
https://github.com/vukrosic/gpu-kernel-engineer-from-scratch

Community:
https://www.skool.com/become-ai-researcher-2669/about

Roadmap:
Month 1 - GPU Foundations
Month 2 - Memory And Benchmarking
Month 3 - Reductions
Month 4 - Scans, Atomics, Synchronization
Month 5 - Softmax And Normalization
Month 6 - Matmul Foundations
Month 7 - Triton For AI Kernels
Month 8 - Triton Matmul And Tuning
Month 9 - PyTorch Integration
Month 10 - Transformer Kernels
Month 11 - Attention And Inference
Month 12 - Portfolio And Interviews
```

## Tags

```text
GPU kernels, CUDA, Triton, AI systems, ML engineering, GPU programming, CUDA tutorial, Triton tutorial, machine learning systems, AI infrastructure, performance engineering, matmul, softmax, attention, FlashAttention, PyTorch, GPU roadmap
```

## Recording Style

Tone:

- ambitious but grounded
- direct and practical
- career-aware but not hype-only
- "I am giving you the map I wish I had"

Visuals:

- repo README
- `course/syllabus.md`
- `weeks/week-01-gpu-mental-model.md`
- simple slides for the 12-month roadmap
- optional whiteboard for CPU vs GPU mental model

Do not live-code too much in this video. Show the repo, show the structure, and
explain the journey.

## 50-Minute Structure

### 0:00-2:30 - Hook

Goal: make the viewer care.

Talking points:

- AI is not only models and papers.
- Modern AI depends on kernels, memory, throughput, and infrastructure.
- Most people can call PyTorch. Fewer can explain what happens underneath.
- This roadmap is for learning the lower-level skill stack behind serious AI systems work.

Sample line:

```text
If you can write, benchmark, and explain GPU kernels, you are much closer to the
systems layer that makes modern AI actually run.
```

### 2:30-5:00 - Who This Is For

Goal: qualify the viewer.

This is for:

- Python programmers who want to go deeper
- ML engineers who use PyTorch but want to understand performance
- CUDA beginners
- AI systems learners
- people building a public technical portfolio

This is not for:

- people who want a weekend shortcut
- people who only want high-level ML theory
- people who refuse to test and benchmark code

### 5:00-8:00 - What You Will Build

Goal: make the destination concrete.

By the end, the portfolio should include:

- CUDA kernels
- Triton kernels
- correctness tests
- benchmark tables
- notes explaining bottlenecks
- matmul, softmax, layer norm, fused activation, attention-style kernels
- final capstone README
- resume bullets and interview explanations

Show the repo structure.

### 8:00-11:00 - How The Course Works

Goal: explain the weekly rhythm.

Core rhythm:

```text
Every week: learn one concept, build one thing, test it, benchmark it, write it down.
```

Explain:

- one flagship video per week
- each week has a top-to-bottom Markdown file
- each month has three build weeks and one checkpoint week
- every assignment has Minimum, Standard, and Stretch
- minimum keeps people moving when life gets busy

Show:

- `weeks/week-01-gpu-mental-model.md`
- `course/recovery-system.md`

### 11:00-15:00 - How Not To Quit

Goal: show that the roadmap is hard but survivable.

Talking points:

- This is a year-long skill tree.
- Falling behind is expected.
- The enemy is not missing one day. The enemy is disappearing for six weeks.
- Do the minimum version when you are busy.
- Checkpoint weeks are part of the plan, not failure weeks.

Sample line:

```text
Correct and finished beats perfect and abandoned.
```

### 15:00-18:00 - Month 1: GPU Foundations

Weeks:

1. GPU mental model and baseline
2. CUDA setup and vector add
3. grids, blocks, threads, indexing
4. checkpoint and Month 1 writeup

Explain:

- start with baseline references
- vector add is simple but teaches indexing
- no GPU yet is okay for Week 1
- first portfolio artifact is a benchmark note

### 18:00-21:00 - Month 2: Memory And Benchmarking

Weeks:

1. global memory bandwidth
2. coalesced vs strided access
3. reliable timing harness
4. memory bandwidth report

Explain:

- memory movement is often the bottleneck
- bad benchmarks create fake confidence
- this month teaches measurement discipline

### 21:00-24:00 - Month 3: Reductions

Weeks:

1. row sum and row max
2. shared-memory reductions
3. warp-level thinking
4. reduction benchmark report

Explain:

- reductions are the first real coordination problem
- many values become fewer values
- this prepares for softmax and normalization

### 24:00-27:00 - Month 4: Scans, Atomics, Synchronization

Weeks:

1. barriers and race conditions
2. atomics and histograms
3. prefix sum / scan
4. synchronization interview notes

Explain:

- parallel code fails in ways normal Python code does not
- synchronization is where mental models get tested
- this month builds debugging maturity

### 27:00-30:00 - Month 5: Softmax And Normalization

Weeks:

1. safe row-wise softmax
2. fused softmax
3. LayerNorm
4. normalization systems note

Explain:

- now the course touches transformer-adjacent kernels
- numerical stability matters
- fusion reduces memory traffic

### 30:00-34:00 - Month 6: Matmul Foundations

Weeks:

1. naive matmul
2. tiled matmul
3. tile sizes and occupancy
4. matmul portfolio page

Explain:

- matmul is the center of deep learning compute
- tiling is about reuse
- this is a major portfolio milestone

### 34:00-37:00 - Month 7: Triton For AI Kernels

Weeks:

1. Triton mental model
2. blocks and masks
3. Triton softmax
4. CUDA vs Triton comparison

Explain:

- CUDA is the mental model
- Triton is the practical AI-kernel tool
- this month makes kernels feel closer to Python ML workflows

### 37:00-40:00 - Month 8: Triton Matmul And Tuning

Weeks:

1. Triton matmul
2. autotuning ideas
3. batched matmul
4. size vs speed chart

Explain:

- tuning is experimental
- compare block sizes, warps, stages
- record results even when they are messy

### 40:00-42:00 - Month 9: PyTorch Integration

Weeks:

1. PyTorch baselines
2. custom op wrapper
3. GPU test matrix
4. installation and demo docs

Explain:

- kernels matter more when they connect to real ML workflows
- make the repo reviewer-friendly
- test shapes, dtypes, and tolerances

### 42:00-44:00 - Month 10: Transformer Kernels

Weeks:

1. GELU and activation fusion
2. RMSNorm
3. attention pieces
4. transformer bottleneck note

Explain:

- connect the kernels to transformer architecture
- show that this is not random CUDA practice
- each kernel maps to a recognizable AI operation

### 44:00-46:00 - Month 11: Attention And Inference

Weeks:

1. attention forward pass
2. FlashAttention concepts
3. KV cache basics
4. capstone draft

Explain:

- attention is a memory problem, not just a math formula
- inference introduces different bottlenecks
- this month becomes the capstone core

### 46:00-48:00 - Month 12: Portfolio And Interviews

Weeks:

1. benchmark dashboard
2. interview explanations
3. resume and project story
4. final capstone

Explain:

- the final month packages the work
- reviewers need to understand what you built, tested, measured, and learned
- resume bullets should come from real work, not vague claims

### 48:00-50:00 - Free Repo And Skool CTA

Goal: convert without sounding desperate.

Free repo:

- roadmap
- assignments
- starter code
- tests
- week files
- portfolio structure

Skool:

- build it with other people
- weekly progress threads
- group office hours
- benchmark comparisons
- portfolio packaging
- interview practice
- demo days

Sample CTA:

```text
If you want to do this alone, the GitHub repo is free. Start with Week 01.

If you want to build it with other people, compare benchmarks, join office
hours, and package the work into a portfolio, join the Skool community.
```

End line:

```text
Start Week 01 today. Do the minimum version if you are busy. The important thing
is that you start building the portfolio.
```

## On-Screen Chapters

```text
00:00 Why GPU kernels matter
02:30 Who this roadmap is for
05:00 What you will build
08:00 How the course works
11:00 How not to quit
15:00 Month 1: GPU Foundations
18:00 Month 2: Memory And Benchmarking
21:00 Month 3: Reductions
24:00 Month 4: Synchronization
27:00 Month 5: Softmax And LayerNorm
30:00 Month 6: Matmul
34:00 Month 7: Triton
37:00 Month 8: Triton Matmul
40:00 Month 9: PyTorch Integration
42:00 Month 10: Transformer Kernels
44:00 Month 11: Attention And Inference
46:00 Month 12: Portfolio And Interviews
48:00 Free repo and community
```

## Filming Checklist

Before recording:

- open the GitHub repo
- open `README.md`
- open `course/syllabus.md`
- open `weeks/week-01-gpu-mental-model.md`
- open `course/recovery-system.md`
- prepare one simple roadmap slide
- prepare one CPU vs GPU visual

During recording:

- keep the pace moving
- do not explain every kernel deeply
- repeat that the weekly files are the actual action plan
- show Week 01 clearly
- end with one action: start Week 01

After publishing:

- pin the GitHub and Skool links
- post the video in Skool
- create a Week 01 progress thread
- make 3 shorts from the recording

## Shorts To Cut From This Video

1. "Most ML engineers use PyTorch. Fewer understand the kernels underneath."
2. "The 12-month GPU kernel roadmap in 60 seconds."
3. "How not to quit a year-long technical course."
4. "CUDA is the mental model. Triton is the AI-kernel tool."
5. "Why matmul is the center of deep learning compute."
6. "Attention is a memory problem."
7. "Correct and finished beats perfect and abandoned."
