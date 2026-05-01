from __future__ import annotations

import time
from typing import Callable

import numpy as np


def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a) + np.asarray(b)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a) @ np.asarray(b)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = q.shape[-1]
    scores = (q @ k.T) / np.sqrt(d)
    weights = softmax(scores, axis=-1)
    return weights @ v


def benchmark(fn: Callable, *args, repeats: int = 100) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    end = time.perf_counter()
    return (end - start) / repeats
