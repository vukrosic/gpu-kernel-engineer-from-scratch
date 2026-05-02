# Critique: GPU Kernel Engineer From Scratch

## Short Verdict

This is a strong course concept with an unusually clear shape: one year, one
artifact per week, CUDA first, Triton later, and portfolio evidence throughout.
The biggest strength is that it does not sell GPU kernels as magic. It teaches
the habits that actually matter: trusted references, correctness checks,
benchmarking, and written explanations.

After the full writing pass, the course now reads like a real curriculum rather
than a scaffold. Every week has a lesson flow, a build target, a code sketch,
and a write-up prompt. The remaining weakness is not the curriculum shape; it is
the lack of executed evidence. A reviewer still cannot trust the repo until the
results pages contain real hardware, real timings, real comparisons, and a final
capstone that has actually been run.

## What Works

The positioning is excellent. "GPU kernel engineer from scratch" is specific,
career-relevant, and easy to understand. It sits in a useful gap between generic
CUDA tutorials and high-level ML courses. The promise is not "learn CUDA syntax";
it is "build a public GPU systems portfolio." That is much sharper.

The 12-month structure is sensible. Starting with mental models and CPU/NumPy
baselines before writing CUDA is the right move. The course then progresses
through memory, reductions, synchronization, softmax, normalization, matmul,
Triton, PyTorch integration, transformer kernels, attention, and portfolio
packaging. That is a credible path from beginner to ML systems fluency.

The weekly rhythm is one of the best parts of the design. Each week asks the
learner to understand, implement, test, benchmark, and write. That rhythm is
more valuable than another pile of lectures. It gives students a repeatable
engineering loop.

The recovery system is humane and strategically correct. GPU programming has a
high frustration ceiling, so the Minimum / Standard / Stretch split is not just
nice pedagogy; it is retention infrastructure. The course explicitly protects
momentum, which matters for a year-long technical path.

The Skool offer is also pointed in the right direction. The free repo gives the
map, while the paid community sells feedback, debugging help, accountability,
portfolio review, and interview practice. That is a clean ethical split. Do not
gate the roadmap; gate the support.

## What Is Weak

The repo still has a proof gap. The curriculum is now written, but the evidence
layer is still mostly template-shaped. For this kind of course, the public
artifact matters as much as the curriculum. A skeptical reviewer should be able
to open `results/` and immediately see real hardware, shapes, baselines,
timings, correctness checks, and limits.

The testing layer is too light for the stated ambition. The current tests mostly
check imports, shapes, simple NumPy behavior, and file existence. That is useful
for scaffolding, but not enough for a GPU kernels portfolio. The course needs
shape grids, dtype coverage, numerical tolerance policy, edge cases,
non-contiguous tensors, skip behavior for missing GPU dependencies, and at least
one serious test matrix that feels like production ML systems work.

The profiling story is underdeveloped. Benchmarking is present, but a GPU kernel
course should eventually teach students how to explain performance with more
than elapsed time. The roadmap should introduce occupancy, memory bandwidth,
roofline intuition, Nsight Systems / Nsight Compute, achieved bandwidth,
arithmetic intensity, kernel launch overhead, and when a custom kernel is
educational rather than better than PyTorch.

The later weeks now read well, but they still need the same level of execution
discipline as the front half. The written lessons are there; the next barrier is
shipping the matching code, tests, benchmark runs, and finished result files so
the back half earns the same trust as the first month.

There is a slight mismatch between "from scratch" and the existing code. Some
Triton kernels already exist as starter implementations, which is fine, but the
course should be explicit about what the learner writes versus what is provided.
Otherwise "from scratch" can feel like a branding phrase instead of a learning
contract.

The capstone needs a stronger spine. "Attention or transformer kernels" is the
right destination, and the last quarter now points there, but the repo should
pick one flagship capstone and make the final quarter unmistakably about it.
For example: PyTorch attention reference, NumPy reference, Triton softmax and
matmul pieces, simplified attention forward, benchmark dashboard, and final
explanation. A single destination will make the year feel less like 48
disconnected exercises.

## Highest-Leverage Fixes

1. Make the current state honest in the README.

   Add a small "Project Status" section that says what is complete, what is
   scaffolded, and what still needs real benchmark evidence. This will increase
   trust rather than weaken the course.

2. Create one gold-standard finished week.

   Week 02 or Week 27 would be ideal. It should include the lesson, code,
   correctness tests, benchmark run, result note, common bugs, and portfolio
   explanation. This becomes the quality bar for every later week.

3. Upgrade tests before adding more prose.

   Add parameterized tests for references and CPU fallbacks first. Then add GPU
   tests that skip cleanly when CUDA/Triton is unavailable. A GPU course without
   serious correctness coverage feels decorative.

4. Add a benchmark schema and generated dashboard.

   Results should not be hand-wavy prose. Use a consistent table with operation,
   implementation, hardware, dtype, shape, baseline, median time, bandwidth or
   TFLOPS where relevant, speedup, and notes. The course promise becomes much
   more credible when every result has the same structure.

5. Introduce profiling as a first-class skill.

   Benchmarking answers "how fast"; profiling answers "why." Add one profiling
   week or profiling appendix with Nsight commands, what metrics to inspect, and
   a short example of interpreting a bottleneck.

6. Strengthen the back half before promotion.

   Months 9-12 are the career payoff: PyTorch integration, transformer kernels,
   attention, benchmark dashboard, interview explanations, and capstone. These
   should not feel like placeholders. They are where the course becomes a
   portfolio instead of a roadmap.

7. Turn the capstone into one named public artifact.

   Recommended capstone: "A simplified transformer attention path from reference
   to Triton." It should include PyTorch and NumPy references, one or more Triton
   kernels, correctness tests, benchmark comparison, profiler notes, limitations,
   and interview-ready explanation.

## Suggested Quality Bar

A week is truly finished only when it has:

- a clear learning objective
- one implementation target
- a reference or trusted baseline
- correctness tests across more than one shape
- a benchmark with hardware and units
- a result note with a lesson learned and limitation
- one portfolio sentence the learner could reuse later

A month is truly finished only when it has:

- a summary table of all kernels or concepts
- at least one real benchmark table
- one hard bug or debugging lesson
- one "what changed from the previous month" explanation
- one public-facing artifact worth showing

The course is truly finished only when a reviewer can clone the repo, run the
CPU path, understand the optional GPU path, inspect real results, and see a
final capstone that ties the year together.

## Marketing Critique

The roadmap video should sell the transformation, not the volume. "48 weeks" is
less compelling than "you will become the kind of engineer who can write a
kernel, prove it correct, benchmark it honestly, and explain the performance."

The strongest hook is:

> Most ML engineers can call PyTorch. Far fewer can explain, test, and improve
> the kernels underneath it.

The course should avoid sounding like another massive study plan. The emotional
promise is not "do 48 assignments." It is "stop feeling blind when performance
matters."

## Final Assessment

This is worth building. The concept is strong, the audience is real, and the
course architecture is better than most technical roadmaps because it includes
recovery, artifacts, and portfolio packaging from the beginning.

The next phase should be less curriculum expansion and more evidence. Finish one
vertical slice all the way: code, tests, benchmark, result note, profiler
interpretation, and portfolio explanation. Once one slice is undeniable, repeat
that standard across the roadmap.

The course should aspire to be judged not by how complete the syllabus looks,
but by whether a learner can point to the repo and say:

> I built this. It is correct. I measured it. I know why it performs this way.
