
#Call using pytest -v matvec_tests.py


import numpy as np
import pytest

from matvec_multiply import dot_product, matrix_vector_product


# ----------------------------------------------------------------------
# Tests for dot_product
# ----------------------------------------------------------------------

def test_dot_product_basic():
    # 1*4 + 2*5 + 3*6 = 32
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32


def test_dot_product_orthogonal_vectors():
    # Orthogonal vectors have a dot product of zero
    assert dot_product([1, 0], [0, 1]) == 0


def test_dot_product_zero_vector():
    assert dot_product([0, 0, 0], [5, -3, 2]) == 0


def test_dot_product_negative_values():
    assert dot_product([-1, 2, -3], [4, -5, 6]) == (-1 * 4) + (2 * -5) + (-3 * 6)


def test_dot_product_single_element():
    assert dot_product([7], [6]) == 42


def test_dot_product_mismatched_length_raises():
    with pytest.raises(ValueError):
        dot_product([1, 2, 3], [1, 2])


def test_dot_product_matches_numpy_random():
    # Cross-check against numpy on random data as a broader correctness test
    rng = np.random.default_rng(0)
    v1 = rng.random(50)
    v2 = rng.random(50)
    assert dot_product(v1.tolist(), v2.tolist()) == pytest.approx(np.dot(v1, v2))


# ----------------------------------------------------------------------
# Tests for matrix_vector_product
# ----------------------------------------------------------------------

def test_matrix_vector_product_basic():
    matrix = [[1, 2], [3, 4]]
    vector = [5, 6]
    # [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
    assert matrix_vector_product(matrix, vector) == [17, 39]


def test_matrix_vector_product_identity():
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    vector = [3, -1, 7]
    # Multiplying by the identity matrix should return the vector unchanged
    assert matrix_vector_product(identity, vector) == vector


def test_matrix_vector_product_zero_matrix():
    matrix = [[0, 0], [0, 0], [0, 0]]
    vector = [4, 9]
    assert matrix_vector_product(matrix, vector) == [0, 0, 0]


def test_matrix_vector_product_single_row():
    matrix = [[1, 2, 3]]
    vector = [1, 1, 1]
    assert matrix_vector_product(matrix, vector) == [6]


def test_matrix_vector_product_dimension_mismatch_raises():
    matrix = [[1, 2, 3], [4, 5, 6]]
    vector = [1, 2]  # wrong length: matrix has 3 columns
    with pytest.raises(ValueError):
        matrix_vector_product(matrix, vector)


def test_matrix_vector_product_matches_numpy_random():
    # Cross-check against numpy's matmul on a random non-square matrix
    rng = np.random.default_rng(1)
    m = rng.random((20, 15))
    v = rng.random(15)
    result = matrix_vector_product(m.tolist(), v.tolist())
    expected = m @ v
    assert result == pytest.approx(expected.tolist())