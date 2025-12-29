"""
SCL - Single Cell Library

High-performance computational kernels for single-cell analysis.

Sparse Matrices:
    CsrMatrix: Compressed Sparse Row (row-oriented operations)
    CscMatrix: Compressed Sparse Column (column-oriented operations)
    SparseMatrix: Alias for CsrMatrix (most common use case)

Dense Matrices:
    DenseMatrix: UNSAFE wrapper for temporary views only

Library Selection:
    Library is automatically selected based on data dtypes.
    Use set_precision() to change global defaults.

Usage:
    >>> import scl
    >>> from scipy.sparse import random as sp_random
    
    # Create CSR matrix (library auto-selected)
    >>> scipy_mat = sp_random(1000, 500, density=0.1, format='csr')
    >>> mat = scl.CsrMatrix.copy(scipy_mat)
    >>> print(mat.shape)
    (1000, 500)
    
    # Row slicing (zero-copy view)
    >>> rows_10_20 = mat[10:20]
    
    # Transpose CSR -> CSC
    >>> csc = mat.T
    
    # Change default precision
    >>> scl.set_precision(real='float32', index='int32')
    >>> mat32 = scl.CsrMatrix.copy(scipy_mat)  # Uses f32_i32
    
    # Dense matrix (UNSAFE - temporary view only)
    >>> import numpy as np
    >>> arr = np.random.randn(1000, 50)
    >>> with scl.DenseMatrix.wrap(arr) as dense:
    ...     # arr MUST stay alive here
    ...     pass
"""

__version__ = "0.1.0"

# Core data structures
from .core import (
    # Sparse matrices
    SparseMatrixBase,
    CsrMatrix,
    CscMatrix,
    SparseMatrix,  # Alias for CsrMatrix
    BlockStrategy,
    OwnershipMode,
    # Dense matrices
    DenseMatrix,
    ArrayView,
    # Enums
    RealType,
    IndexType,
    # Error handling
    SCLError,
    check_error,
    get_last_error,
    clear_error,
    # Configuration
    set_precision,
    get_precision,
    get_library,
    get_default_library,
)

__all__ = [
    # Version
    "__version__",
    # Sparse matrices
    "SparseMatrixBase",
    "CsrMatrix",
    "CscMatrix",
    "SparseMatrix",
    "BlockStrategy",
    "OwnershipMode",
    # Dense matrices
    "DenseMatrix",
    "ArrayView",
    # Enums
    "RealType",
    "IndexType",
    # Error handling
    "SCLError",
    "check_error",
    "get_last_error",
    "clear_error",
    # Configuration
    "set_precision",
    "get_precision",
    "get_library",
    "get_default_library",
]
