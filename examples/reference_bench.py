from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gputriton.bench import run_reference_benchmarks


def main():
    with np.errstate(all="ignore"):
        results = run_reference_benchmarks()
    for name, seconds in results.items():
        print(f"{name}: {seconds:.6f}s")


if __name__ == "__main__":
    main()
