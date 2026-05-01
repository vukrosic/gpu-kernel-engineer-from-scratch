from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def slug_for_week(n: int) -> str:
    special = {
        1: "baseline",
        8: "month-02-checkpoint",
        12: "month-03-checkpoint",
        16: "month-04-checkpoint",
        20: "month-05-checkpoint",
        24: "month-06-checkpoint",
        28: "month-07-checkpoint",
        32: "month-08-checkpoint",
        36: "month-09-checkpoint",
        40: "month-10-checkpoint",
        44: "month-11-checkpoint",
        48: "final-capstone",
    }
    if n in special:
        return special[n]

    titles = {
        2: "vector-add",
        3: "indexing",
        4: "month-01-checkpoint",
        5: "memory-bandwidth",
        6: "coalescing",
        7: "timing-harness",
        9: "naive-reductions",
        10: "shared-reductions",
        11: "warp-thinking",
        13: "synchronization",
        14: "atomics-histograms",
        15: "scan",
        17: "softmax-math",
        18: "fused-softmax",
        19: "layernorm",
        21: "naive-matmul",
        22: "tiled-matmul",
        23: "tiling-occupancy",
        25: "triton-mental-model",
        26: "triton-blocks-masks",
        27: "triton-softmax",
        29: "triton-matmul",
        30: "autotuning",
        31: "batched-matmul",
        33: "pytorch-baselines",
        34: "custom-op-wrapper",
        35: "gpu-test-matrix",
        37: "gelu-fusion",
        38: "rmsnorm",
        39: "attention-pieces",
        41: "attention-forward",
        42: "flashattention-concepts",
        43: "kv-cache",
        45: "benchmark-dashboard",
        46: "interview-explanations",
        47: "resume-and-story",
    }
    return titles[n]


def week_title(n: int) -> str:
    if n == 1:
        return "Week 01 Baseline"
    if n == 48:
        return "Week 48 Final Capstone"
    return f"Week {n:02d}"


def render_week(n: int) -> str:
    return f"""# {week_title(n)}

Status: scaffolded

## What Was Built

TBD

## Correctness Check

TBD

## Benchmark Or Observation

TBD

## Lesson Learned

TBD

## Limitation Or Next Step

TBD
"""


def render_month(n: int) -> str:
    return f"""# Month {n:02d} Checkpoint

Status: scaffolded

## Summary

TBD

## Best Result

TBD

## Hardest Bug

TBD

## Interview Explanation

TBD

## Portfolio Link

TBD
"""


def render_generic(title: str) -> str:
    return f"""# {title}

Status: scaffolded

## What Was Built

TBD

## Correctness Check

TBD

## Benchmark Or Observation

TBD

## Lesson Learned

TBD
"""


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    for n in range(1, 49):
        path = RESULTS / f"week-{n:02d}-{slug_for_week(n)}.md"
        if not path.exists():
            path.write_text(render_week(n), encoding="utf-8")

    for n in range(1, 13):
        path = RESULTS / f"month-{n:02d}-checkpoint.md"
        if not path.exists():
            path.write_text(render_month(n), encoding="utf-8")

    extras = {
        "benchmark-dashboard.md": render_generic("Benchmark Dashboard"),
        "gpu-test-matrix.md": render_generic("GPU Test Matrix"),
    }
    for name, contents in extras.items():
        path = RESULTS / name
        if not path.exists():
            path.write_text(contents, encoding="utf-8")

    figures = RESULTS / "figures"
    figures.mkdir(exist_ok=True)
    gitkeep = figures / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
