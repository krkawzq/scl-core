"""
SCL Core - High-level Python interface for SCL operations.

Provides:
- CsrMatrix: CSR format (row-oriented, fast row slicing)
- CscMatrix: CSC format (column-oriented, fast column slicing)
- DenseMatrix: Unsafe dense matrix wrapper (temporary views only)
- Automatic library selection based on data types

Usage:
    >>> import scl
    >>> from scl.core import CsrMatrix, CscMatrix
    
    # Library auto-selected from data
    >>> mat = CsrMatrix.copy(scipy_matrix)
    
    # Row slicing (zero-copy)
    >>> rows = mat[10:20]
    
    # Transpose CSR -> CSC
    >>> csc = mat.T
    
    # Change default precision
    >>> scl.set_precision(real='float32', index='int32')
"""

from .error import (
    SCLError,
    check_error,
    get_last_error,
    clear_error,
    # Error codes
    SCL_OK,
    SCL_ERROR_UNKNOWN,
    SCL_ERROR_INTERNAL,
    SCL_ERROR_OUT_OF_MEMORY,
    SCL_ERROR_NULL_POINTER,
    SCL_ERROR_INVALID_ARGUMENT,
    SCL_ERROR_DIMENSION_MISMATCH,
    SCL_ERROR_NOT_IMPLEMENTED,
)

from .config import (
    RealType,
    IndexType,
    set_precision,
    get_precision,
    get_library,
    get_default_library,
)

from .sparse import (
    SparseMatrixBase,
    CsrMatrix,
    CscMatrix,
    SparseMatrix,  # Alias for CsrMatrix
    BlockStrategy,
    OwnershipMode,
)

from .dense import (
    DenseMatrix,
    ArrayView,
)

__all__ = [
    # Error handling
    "SCLError",
    "check_error",
    "get_last_error",
    "clear_error",
    "SCL_OK",
    "SCL_ERROR_UNKNOWN",
    "SCL_ERROR_INTERNAL",
    "SCL_ERROR_OUT_OF_MEMORY",
    "SCL_ERROR_NULL_POINTER",
    "SCL_ERROR_INVALID_ARGUMENT",
    "SCL_ERROR_DIMENSION_MISMATCH",
    "SCL_ERROR_NOT_IMPLEMENTED",
    # Configuration
    "RealType",
    "IndexType",
    "set_precision",
    "get_precision",
    "get_library",
    "get_default_library",
    # Sparse matrices
    "SparseMatrixBase",
    "CsrMatrix",
    "CscMatrix",
    "SparseMatrix",  # Alias
    "BlockStrategy",
    "OwnershipMode",
    # Dense matrices
    "DenseMatrix",
    "ArrayView",
]
