import numpy as np

from gputriton import attention, matmul, softmax, vector_add


def test_vector_add():
    assert np.allclose(vector_add(np.array([1, 2]), np.array([3, 4])), np.array([4, 6]))


def test_softmax_normalizes():
    probs = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert np.allclose(probs.sum(axis=1), np.array([1.0]))


def test_attention_shape():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(4, 8))
    k = rng.normal(size=(4, 8))
    v = rng.normal(size=(4, 8))
    out = attention(q, k, v)
    assert out.shape == (4, 8)


def test_matmul_shape():
    a = np.ones((2, 3))
    b = np.ones((3, 4))
    assert matmul(a, b).shape == (2, 4)
