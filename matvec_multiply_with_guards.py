
import random
import time

import numpy as np


# ----------------------------------------------------------------------
# Version 1: pure Python, explicit for-loops
# ----------------------------------------------------------------------

def dot_product(vector1, vector2):
    """
    Compute the dot product of two vectors using a for loop.

    Parameters
    ----------
    vector1 : list of float
        First input vector.
    vector2 : list of float
        Second input vector, must be the same length as vector1.

    Returns
    -------
    float
        The scalar dot product of vector1 and vector2.

    Raises
    ------
    TypeError
        If either input is not a list/tuple, or contains a non-numeric element.
    ValueError
        If either vector is empty, or the two vectors have different lengths.
    """
    # Guard: inputs must be list-like, not e.g. a scalar or None
    if not isinstance(vector1, (list, tuple)) or not isinstance(vector2, (list, tuple)):
        raise TypeError("Both inputs to dot_product must be lists or tuples")

    # Guard: reject empty vectors rather than silently returning 0.0
    if len(vector1) == 0 or len(vector2) == 0:
        raise ValueError("Vectors must not be empty")

    # Guard: lengths must match, with the actual sizes in the error message
    if len(vector1) != len(vector2):
        raise ValueError(
            f"Vectors must be of the same length (got {len(vector1)} and {len(vector2)})"
        )

    result = 0.0
    # Accumulate the sum of element-wise products
    for i in range(len(vector1)):
        a, b = vector1[i], vector2[i]
        # Guard: catch non-numeric elements explicitly, since e.g. "a" * 3
        # would silently repeat a string instead of raising an error
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError(
                f"Vector elements must be numeric (index {i}: {type(a).__name__}, {type(b).__name__})"
            )
        result += a * b

    return result


def matrix_vector_product(matrix, vector):
    """
    Compute the matrix-vector product using dot_product.

    Parameters
    ----------
    matrix : list of list of float
        An (m x n) matrix, represented as a list of m row vectors.
    vector : list of float
        An n-dimensional vector.

    Returns
    -------
    list of float
        The resulting m-dimensional vector.

    Raises
    ------
    TypeError
        If matrix/vector are not list-like, or a row of the matrix isn't
        list-like.
    ValueError
        If the matrix is empty, its rows have inconsistent lengths (ragged),
        or the column count doesn't match the vector length.
    """
    # Guard: inputs must be list-like, not e.g. a scalar or None
    if not isinstance(matrix, (list, tuple)) or not isinstance(vector, (list, tuple)):
        raise TypeError("matrix must be a list of rows and vector must be a list")

    # Guard: an empty matrix has no rows to index into
    if len(matrix) == 0:
        raise ValueError("Matrix must not be empty")

    # Guard: every row must itself be list-like (catches a flat 1-D matrix)
    if not all(isinstance(row, (list, tuple)) for row in matrix):
        raise TypeError("Each row of the matrix must be a list or tuple")

    # Guard: all rows must be the same length (reject a ragged matrix
    # upfront, rather than failing partway through the loop below)
    row_length = len(matrix[0])
    if any(len(row) != row_length for row in matrix):
        raise ValueError("All rows of the matrix must have the same length")

    # Guard: column count must match vector length, with actual sizes shown
    if row_length != len(vector):
        raise ValueError(
            f"Matrix column count ({row_length}) must match vector length ({len(vector)})"
        )

    result = []
    # Each output entry is the dot product of one matrix row with the vector
    for row in matrix:
        result.append(dot_product(row, vector))

    return result


# ----------------------------------------------------------------------
# Version 2: vectorized with numpy (no explicit Python-level loop)
# ----------------------------------------------------------------------

def dot_product_np(vector1, vector2):
    """
    Compute the dot product of two vectors using numpy's vectorized np.dot.

    Parameters
    ----------
    vector1 : np.ndarray
        First input vector.
    vector2 : np.ndarray
        Second input vector, must be the same length as vector1.

    Returns
    -------
    float
        The scalar dot product of vector1 and vector2.
    """
    # np.dot pushes the element-wise multiply + sum into compiled C code
    return np.dot(vector1, vector2)


def matrix_vector_product_np(matrix, vector):
    """
    Compute the matrix-vector product using numpy's vectorized matmul.

    Parameters
    ----------
    matrix : np.ndarray
        An (m x n) matrix.
    vector : np.ndarray
        An n-dimensional vector.

    Returns
    -------
    np.ndarray
        The resulting m-dimensional vector.
    """
    # matrix @ vector applies dot_product_np across all rows at once
    return matrix @ vector


def main():
    """
    Run both versions of matrix-vector product on the same randomly
    generated 1000x1000 matrix and 1000-element vector, then print a
    side-by-side comparison of correctness and runtime.
    """
    n = 1000

    # Generate the data once as numpy arrays, then derive a list-of-lists
    # copy so both versions operate on identical values
    matrix_np = np.random.rand(n, n)
    vector_np = np.random.rand(n)
    matrix_list = matrix_np.tolist()
    vector_list = vector_np.tolist()

    # --- Version 1: pure Python for-loop ---
    start = time.perf_counter()
    result_loop = matrix_vector_product(matrix_list, vector_list)
    time_loop = time.perf_counter() - start

    # --- Version 2: numpy vectorized ---
    start = time.perf_counter()
    result_np = matrix_vector_product_np(matrix_np, vector_np)
    time_np = time.perf_counter() - start

    # Sanity check: both versions should agree (within floating-point error)
    max_diff = np.max(np.abs(np.array(result_loop) - result_np))

    print(f"{'':20s}{'Pure Python loop':>20s}{'Numpy vectorized':>20s}")
    print(f"{'Time (seconds)':20s}{time_loop:>20.4f}{time_np:>20.4f}")
    print(f"{'Speedup (x)':20s}{'1.0':>20s}{time_loop / time_np:>20.1f}")
    print(f"\nMax difference between results: {max_diff:.2e}")
    print(f"First 5 entries (loop):  {result_loop[:5]}")
    print(f"First 5 entries (numpy): {result_np[:5]}")


if __name__ == "__main__":
    main()