# Results

This folder holds the proof that the course is real: benchmark notes, monthly
checkpoints, benchmark dashboards, and the final capstone writeup.

## File Map

- Weekly notes: `week-01-...md` through `week-48-...md`
- Monthly checkpoints: `month-01-checkpoint.md` through
  `month-12-checkpoint.md`
- Dashboard files: `benchmark-dashboard.md`, `gpu-test-matrix.md`
- Charts and figures: `figures/`

Use `make bootstrap-results` to regenerate the scaffold if you add more weeks or
want to rebuild the template set.

## Benchmark Entry Template

```text
Kernel:
Implementation:
Hardware:
Software:
Input shapes:
Baseline:
Result:
Speedup:
Notes:
```

## Rules

- Include hardware details.
- Include input shapes.
- Include the baseline.
- Include units.
- Do not report speedups without explaining the comparison.
