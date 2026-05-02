# Week Writing Plan

## Goal

Turn all 48 week files into complete, followable lessons. The immediate target
is not to finish every kernel implementation. The target is to make every week
clear enough that a learner can open the file, know what to read, what to run,
what to build, what to test, what to benchmark, what to write down, and what
counts as done.

## Current State

- Week 01 is the quality reference: detailed, concrete, and followable.
- Weeks 02-10 are partially developed and need expansion, consistency, and
  stronger result-writing prompts.
- Weeks 11-48 are mostly thin lesson scaffolds and need full writing.
- The results files are scaffolded separately. They should be improved after the
  week files are written, not before.

## Writing Contract For Every Week

Each weekly lesson should include:

- what the week is about and why it matters
- the exact prerequisite from the previous week
- 1-3 files to read, not a broad reading dump
- exact commands to run
- one concrete build target
- one correctness check
- one benchmark or measured observation
- one result note path under `results/`
- 3-5 explanation prompts
- Minimum, Standard, and Stretch paths
- what to skip if overwhelmed
- a preview of the next week

Each week should avoid:

- vague "study this topic" language
- pretending a GPU is available when CPU-only learners can still do setup work
- benchmark claims without hardware, shape, baseline, and units
- asking learners to optimize before correctness is established
- adding unrelated theory that does not support the week's artifact

## Definition Of A Fully Written Week

A week is complete when a learner can do the Standard path without guessing:

1. Run the listed command or know why it may skip.
2. Build or edit the named artifact.
3. Compare output against a trusted baseline.
4. Record a benchmark, profiler observation, or measured result.
5. Write a short note in the named result file.
6. Explain the core idea in plain language.

## Authoring Order

### Phase 1: Establish The Gold Standard

Write Week 02 fully first, then revise Week 01 only if the new structure reveals
small consistency fixes.

Why Week 02:

- It is the first real CUDA week.
- It sets the tone for setup, GPU availability, correctness, and benchmark
  expectations.
- It creates the pattern for future implementation weeks.

Deliverables:

- fully expanded `weeks/week-02-gpu-setup-and-vector-add.md`
- stronger result prompts for `results/week-02-vector-add.md`
- any README pointer needed to clarify CPU-only versus CUDA paths

### Phase 2: Finish Month 1

Complete Weeks 03-04.

Week 03 should become the elementwise kernel practice week:

- add, multiply, square, and ReLU
- index calculation
- shape handling
- CPU reference comparison
- note on why elementwise kernels are simple but foundational

Week 04 should become the first checkpoint model:

- rerun tests
- package Month 1 results
- explain grids, blocks, threads, and indexing from memory
- write a Month 1 portfolio paragraph

Deliverable:

- Month 1 should be the first fully usable month of the course.

### Phase 3: Finish Months 2-3

Complete Weeks 05-12.

Month 2 focus:

- memory bandwidth
- coalesced versus strided access
- timing discipline
- benchmark reliability

Month 3 focus:

- reductions
- shared memory
- warp-level intuition
- reduction checkpoint

Special care:

- Week 07 should teach benchmark hygiene clearly: warmup, repeats, median,
  synchronization, and why naive timing lies.
- Week 11 should stay conceptual enough for beginners but still tied to a
  concrete reduction artifact.

Deliverable:

- first quarter complete enough to publish as an initial course slice.

### Phase 4: Finish CUDA Fundamentals

Complete Weeks 13-24.

Month 4 focus:

- synchronization
- race conditions
- atomics
- histograms
- prefix sum / scan

Month 5 focus:

- safe softmax
- fused softmax
- LayerNorm
- numerical stability

Month 6 focus:

- naive matmul
- tiled matmul
- occupancy and tile-size reasoning
- matmul checkpoint

Special care:

- These weeks are where learners can get lost. Each lesson should name the
  smallest correct version first.
- Softmax and LayerNorm weeks need explicit numerical tolerance guidance.
- Matmul weeks need diagrams in words: rows, columns, tiles, and shared memory.

Deliverable:

- CUDA half of the course feels complete even before Triton starts.

### Phase 5: Finish Triton Track

Complete Weeks 25-32.

Month 7 focus:

- Triton mental model
- blocks and masks
- Triton softmax
- CUDA versus Triton explanation

Month 8 focus:

- Triton matmul
- autotuning
- batched matmul
- benchmark tables

Special care:

- Be explicit about what Triton abstracts away and what it does not.
- Every Triton week should include CPU fallback expectations and GPU skip notes.
- Autotuning should be taught as disciplined search, not random knob turning.

Deliverable:

- a complete modern AI-kernel section that connects CUDA fundamentals to Triton.

### Phase 6: Finish Integration And Testing

Complete Weeks 33-36.

Focus:

- PyTorch baselines
- clean Python API
- custom op wrapper
- shape, dtype, device, tolerance, and error-case test matrix
- installation and demo docs

Special care:

- Week 35 should become the testing philosophy anchor for the course.
- This month should make the repo feel credible to ML infrastructure reviewers.

Deliverable:

- learners can explain how educational kernels relate to PyTorch production
  baselines.

### Phase 7: Finish Transformer And Attention Arc

Complete Weeks 37-44.

Month 10 focus:

- GELU / activation fusion
- RMSNorm
- attention pieces
- transformer bottleneck explanation

Month 11 focus:

- simplified attention forward
- FlashAttention concepts
- KV cache basics
- capstone draft

Special care:

- This arc should pull toward one named capstone: a simplified transformer
  attention path from reference to Triton.
- FlashAttention should be conceptual and honest. Do not imply the learner has
  rebuilt full production FlashAttention unless the code really does that.
- KV cache should focus on inference memory movement and shape discipline.

Deliverable:

- the course has a strong final technical story instead of isolated kernels.

### Phase 8: Finish Portfolio Month

Complete Weeks 45-48.

Focus:

- benchmark dashboard
- interview explanations
- resume/project story
- final capstone writeup

Special care:

- Week 45 should define the final result table schema.
- Week 46 should produce answers a learner can say out loud.
- Week 47 should turn work into honest resume bullets.
- Week 48 should package the final artifact and next-step roadmap.

Deliverable:

- the course ends with a portfolio, not just a completed checklist.

## Month-By-Month Writing Targets

| Month | Weeks | Writing Target |
| --- | --- | --- |
| 1 | 01-04 | First complete beginner path from setup to first CUDA artifact |
| 2 | 05-08 | Memory and benchmarking discipline |
| 3 | 09-12 | Reductions and first quarter checkpoint |
| 4 | 13-16 | Synchronization, atomics, scans |
| 5 | 17-20 | Softmax, fusion, LayerNorm |
| 6 | 21-24 | Matmul foundations |
| 7 | 25-28 | Triton entry and softmax |
| 8 | 29-32 | Triton matmul and tuning |
| 9 | 33-36 | PyTorch integration and GPU test matrix |
| 10 | 37-40 | Transformer-adjacent kernels |
| 11 | 41-44 | Attention, FlashAttention concepts, KV cache |
| 12 | 45-48 | Dashboard, interviews, resume, capstone |

## Practical Work Rhythm

Use this loop for each batch of four weeks:

1. Read the month page and all four week files.
2. Decide the single month-level artifact.
3. Rewrite the three build weeks.
4. Rewrite the checkpoint week last.
5. Check links and result filenames.
6. Run a quick text audit for placeholder language.

Batch acceptance check:

```bash
rg -n "TBD|Sketch the smallest|Build This$|Status: scaffolded|TODO|placeholder" weeks/week-XX*.md
```

Then manually read each week top to bottom.

## Naming And Link Rules

- Keep existing week filenames.
- Keep existing result filenames unless there is a strong reason to rename them.
- Each week should link to its result file by exact path.
- Each checkpoint week should link to the previous three build weeks and the
  monthly checkpoint result file.
- Avoid adding new directories during the writing pass unless a week truly needs
  a new support file.

## Proposed Immediate Next Step

Start with Week 02 and make it the model implementation lesson.

After Week 02 is complete, write Weeks 03 and 04 in the same style. Then pause
and compare Month 1 against this plan before moving to Month 2.
