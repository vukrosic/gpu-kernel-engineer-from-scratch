from pathlib import Path

import numpy as np

import triton_kernels as tk


def test_triton_kernels_importable_without_gpu():
    assert isinstance(tk.gpu_stack_ready(), bool)
    assert isinstance(tk.has_torch(), bool)
    assert isinstance(tk.has_triton(), bool)


def test_vector_add_reference_matches_numpy():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    assert np.allclose(tk.vector_add_reference(a, b), np.array([5.0, 7.0, 9.0], dtype=np.float32))
    assert np.allclose(tk.vector_add(a, b), np.array([5.0, 7.0, 9.0], dtype=np.float32))


def test_softmax_reference_normalizes_rows():
    probs = tk.softmax_reference(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert np.allclose(probs.sum(axis=1), np.array([1.0]))


def test_matmul_reference_matches_numpy_shape():
    a = np.ones((2, 3), dtype=np.float32)
    b = np.ones((3, 4), dtype=np.float32)
    assert tk.matmul_reference(a, b).shape == (2, 4)


def test_kernel_files_exist():
    root = Path(__file__).resolve().parents[1]
    expected = [
        root / "cuda" / "vector_add.cu",
        root / "cuda" / "reduce_sum.cu",
        root / "cuda" / "softmax.cu",
        root / "cuda" / "naive_matmul.cu",
        root / "cuda" / "layernorm.cu",
        root / "triton_kernels" / "vector_add.py",
        root / "triton_kernels" / "reduce_sum.py",
        root / "triton_kernels" / "softmax.py",
        root / "triton_kernels" / "matmul.py",
        root / "triton_kernels" / "layernorm.py",
    ]
    for path in expected:
        assert path.exists(), path
