"""
SCL Math Module.

This module provides mathematical operations for sparse matrices, including:

    - Linear algebra operations (matrix-vector, matrix-matrix products)
    - Element-wise transforms (log, exp, softmax)
    - Correlation and distance metrics

All functions support multiple input formats through the overload dispatch
system, including native SCL types, scipy sparse matrices, and numpy arrays.

Example:
    >>> import scl.math as smath
    >>> from scl import SclCSR
    >>>
    >>> mat = SclCSR.from_scipy(scipy_matrix)
    >>> transformed = smath.log1p(mat)
    >>> correlation = smath.pearson(mat)
"""

from scl.math.linalg import (
    spmv,
    dot,
    gram,
    pearson,
)

from scl.math.transforms import (
    log1p,
    log2p1,
    expm1,
    softmax,
)

__all__ = [
    # Linear algebra
    "spmv",
    "dot",
    "gram",
    "pearson",
    # Transforms
    "log1p",
    "log2p1",
    "expm1",
    "softmax",
]
