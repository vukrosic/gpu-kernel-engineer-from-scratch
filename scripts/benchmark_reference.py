from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gputriton.bench import run_reference_benchmarks


def format_table(results: dict[str, float]) -> str:
    rows = ["| Kernel | Seconds / run |", "| --- | ---: |"]
    for name, seconds in results.items():
        rows.append(f"| {name} | {seconds:.6f} |")
    return "\n".join(rows)


def main() -> None:
    with np.errstate(all="ignore"):
        results = run_reference_benchmarks()
    print("# Reference Benchmarks")
    print()
    print(format_table(results))


if __name__ == "__main__":
    main()
