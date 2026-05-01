from __future__ import annotations

import numpy as np

from .reference import attention, benchmark, matmul, softmax, vector_add


def run_reference_benchmarks() -> dict[str, float]:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(256,))
    b = rng.normal(size=(256,))
    x = rng.normal(size=(64, 64))
    y = rng.normal(size=(64, 64))
    q = rng.normal(size=(32, 64))
    k = rng.normal(size=(32, 64))
    v = rng.normal(size=(32, 64))

    return {
        "vector_add": benchmark(vector_add, a, b, repeats=500),
        "matmul": benchmark(matmul, x, y, repeats=50),
        "softmax": benchmark(softmax, x, repeats=100),
        "attention": benchmark(attention, q, k, v, repeats=20),
    }
