"""
SCL Type Definitions and Protocols.

This module provides type aliases, protocols, and utility functions for
multi-format sparse matrix operations. It enables transparent handling of:

    - Native SCL types (SclCSR, SclCSC)
    - SciPy sparse matrices (csr_matrix, csc_matrix)
    - NumPy arrays (ndarray)
    - Python sequences (List, Tuple)

The dispatch system automatically converts inputs to the appropriate format
and handles copy semantics consistently across all backends.

Example:
    >>> from scl._typing import SparseInput, ensure_scl_csr
    >>>
    >>> def my_func(mat: SparseInput) -> SclCSR:
    ...     csr = ensure_scl_csr(mat, copy=False)
    ...     # ... operations ...
    ...     return result
"""

from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

if TYPE_CHECKING:
    import numpy as np
    from scipy import sparse as sp
    from scl.sparse import SclCSR, SclCSC
    from scl.array import RealArray, IndexArray, ByteArray


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
ArrayT = TypeVar("ArrayT", bound="ArrayLike")
SparseT = TypeVar("SparseT", bound="SparseLike")


# =============================================================================
# Protocol Definitions
# =============================================================================

@runtime_checkable
class ArrayLike(Protocol):
    """Protocol for array-like objects.

    Any object implementing __getitem__, __len__, and shape property
    can be used as an array-like input.
    """

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the array."""
        ...

    def __getitem__(self, key: Any) -> Any:
        """Element access."""
        ...

    def __len__(self) -> int:
        """Length of first dimension."""
        ...


@runtime_checkable
class SparseLike(Protocol):
    """Protocol for sparse matrix-like objects.

    Objects must provide CSR/CSC-compatible interface with:
    - data: Non-zero values
    - indices: Column (CSR) or row (CSC) indices
    - indptr: Row (CSR) or column (CSC) pointers
    - shape: (rows, cols) tuple
    """

    @property
    def data(self) -> Any:
        """Non-zero values array."""
        ...

    @property
    def indices(self) -> Any:
        """Secondary indices array."""
        ...

    @property
    def indptr(self) -> Any:
        """Primary pointer array."""
        ...

    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape (rows, cols)."""
        ...

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        ...


@runtime_checkable
class CSRLike(SparseLike, Protocol):
    """Protocol for CSR matrix-like objects.

    Compressed Sparse Row format where:
    - indices contains column indices
    - indptr contains row pointers
    """
    pass


@runtime_checkable
class CSCLike(SparseLike, Protocol):
    """Protocol for CSC matrix-like objects.

    Compressed Sparse Column format where:
    - indices contains row indices
    - indptr contains column pointers
    """
    pass


# =============================================================================
# Type Aliases
# =============================================================================

# Dense array inputs
DenseInput = Union[
    "np.ndarray",
    Sequence[Sequence[float]],
    List[List[float]],
]

# Sparse matrix inputs
SparseInput = Union[
    "SclCSR",
    "SclCSC",
    "sp.csr_matrix",
    "sp.csc_matrix",
    "sp.spmatrix",
    DenseInput,
]

# CSR-specific inputs
CSRInput = Union[
    "SclCSR",
    "sp.csr_matrix",
]

# CSC-specific inputs
CSCInput = Union[
    "SclCSC",
    "sp.csc_matrix",
]

# Vector inputs
VectorInput = Union[
    "RealArray",
    "np.ndarray",
    Sequence[float],
    List[float],
]

# Index vector inputs
IndexInput = Union[
    "IndexArray",
    "np.ndarray",
    Sequence[int],
    List[int],
]

# Boolean mask inputs
MaskInput = Union[
    "ByteArray",
    "np.ndarray",
    Sequence[bool],
    List[bool],
]

# Output array types
ArrayOutput = Union["RealArray", "np.ndarray"]
IndexOutput = Union["IndexArray", "np.ndarray"]


# =============================================================================
# Format Detection
# =============================================================================

def is_scl_csr(obj: Any) -> bool:
    """Check if object is a native SclCSR matrix.

    Args:
        obj: Object to check.

    Returns:
        True if obj is SclCSR instance.
    """
    # Avoid circular import
    try:
        from scl.sparse import SclCSR
        return isinstance(obj, SclCSR)
    except ImportError:
        return False


def is_scl_csc(obj: Any) -> bool:
    """Check if object is a native SclCSC matrix.

    Args:
        obj: Object to check.

    Returns:
        True if obj is SclCSC instance.
    """
    try:
        from scl.sparse import SclCSC
        return isinstance(obj, SclCSC)
    except ImportError:
        return False


def is_scipy_csr(obj: Any) -> bool:
    """Check if object is a scipy CSR matrix.

    Args:
        obj: Object to check.

    Returns:
        True if obj is scipy.sparse.csr_matrix.
    """
    try:
        from scipy import sparse as sp
        return sp.isspmatrix_csr(obj)
    except ImportError:
        return False


def is_scipy_csc(obj: Any) -> bool:
    """Check if object is a scipy CSC matrix.

    Args:
        obj: Object to check.

    Returns:
        True if obj is scipy.sparse.csc_matrix.
    """
    try:
        from scipy import sparse as sp
        return sp.isspmatrix_csc(obj)
    except ImportError:
        return False


def is_scipy_sparse(obj: Any) -> bool:
    """Check if object is any scipy sparse matrix.

    Args:
        obj: Object to check.

    Returns:
        True if obj is scipy.sparse matrix.
    """
    try:
        from scipy import sparse as sp
        return sp.issparse(obj)
    except ImportError:
        return False


def is_numpy_array(obj: Any) -> bool:
    """Check if object is a numpy ndarray.

    Args:
        obj: Object to check.

    Returns:
        True if obj is numpy.ndarray.
    """
    try:
        import numpy as np
        return isinstance(obj, np.ndarray)
    except ImportError:
        return False


def get_format(obj: Any) -> str:
    """Detect the format of a sparse/dense matrix.

    Args:
        obj: Matrix object.

    Returns:
        Format string: 'scl_csr', 'scl_csc', 'scipy_csr', 'scipy_csc',
        'scipy_other', 'numpy', 'sequence', or 'unknown'.
    """
    if is_scl_csr(obj):
        return "scl_csr"
    elif is_scl_csc(obj):
        return "scl_csc"
    elif is_scipy_csr(obj):
        return "scipy_csr"
    elif is_scipy_csc(obj):
        return "scipy_csc"
    elif is_scipy_sparse(obj):
        return "scipy_other"
    elif is_numpy_array(obj):
        return "numpy"
    elif isinstance(obj, (list, tuple)):
        return "sequence"
    else:
        return "unknown"


# =============================================================================
# Conversion Functions
# =============================================================================

def ensure_scl_csr(
    mat: SparseInput,
    copy: bool = False,
    dtype: str = "float64",
) -> "SclCSR":
    """Convert any sparse input to SclCSR.

    This function provides a unified interface for converting various sparse
    matrix formats to the native SclCSR type. It handles:

    - SclCSR: Returns as-is or copies if requested
    - SclCSC: Converts to CSR format
    - scipy.sparse.csr_matrix: Wraps data arrays
    - scipy.sparse.csc_matrix: Converts to CSR first
    - numpy.ndarray: Creates sparse from dense
    - Sequence: Creates sparse from nested lists

    Args:
        mat: Input matrix in any supported format.
        copy: If True, always create a copy of the data. If False,
            may share data with input when possible.
        dtype: Target data type for values.

    Returns:
        SclCSR matrix containing the same data.

    Raises:
        TypeError: If input format is not supported.
        ValueError: If input has invalid dimensions.

    Example:
        >>> import scipy.sparse as sp
        >>> scipy_mat = sp.random(100, 50, density=0.1, format='csr')
        >>> scl_mat = ensure_scl_csr(scipy_mat)
        >>> print(scl_mat.shape)  # (100, 50)
    """
    from scl.sparse import SclCSR, SclCSC

    fmt = get_format(mat)

    if fmt == "scl_csr":
        if copy:
            return mat.copy()
        return mat

    elif fmt == "scl_csc":
        # CSC -> CSR requires conversion (always creates new data)
        return mat.to_csr()

    elif fmt == "scipy_csr":
        return SclCSR.from_scipy(mat)

    elif fmt in ("scipy_csc", "scipy_other"):
        # Convert to CSR first
        from scipy import sparse as sp
        csr_mat = mat.tocsr()
        return SclCSR.from_scipy(csr_mat)

    elif fmt == "numpy":
        # Dense to sparse
        return _dense_to_csr(mat, dtype)

    elif fmt == "sequence":
        return _dense_to_csr(mat, dtype)

    else:
        raise TypeError(
            f"Cannot convert {type(mat).__name__} to SclCSR. "
            f"Supported types: SclCSR, SclCSC, scipy.sparse, numpy.ndarray, "
            f"List[List[float]]"
        )


def ensure_scl_csc(
    mat: SparseInput,
    copy: bool = False,
    dtype: str = "float64",
) -> "SclCSC":
    """Convert any sparse input to SclCSC.

    This function provides a unified interface for converting various sparse
    matrix formats to the native SclCSC type.

    Args:
        mat: Input matrix in any supported format.
        copy: If True, always create a copy of the data.
        dtype: Target data type for values.

    Returns:
        SclCSC matrix containing the same data.

    Raises:
        TypeError: If input format is not supported.
    """
    from scl.sparse import SclCSR, SclCSC

    fmt = get_format(mat)

    if fmt == "scl_csc":
        if copy:
            return mat.copy()
        return mat

    elif fmt == "scl_csr":
        return mat.to_csc()

    elif fmt == "scipy_csc":
        return SclCSC.from_scipy(mat)

    elif fmt in ("scipy_csr", "scipy_other"):
        from scipy import sparse as sp
        csc_mat = mat.tocsc()
        return SclCSC.from_scipy(csc_mat)

    elif fmt == "numpy":
        csr = _dense_to_csr(mat, dtype)
        return csr.to_csc()

    elif fmt == "sequence":
        csr = _dense_to_csr(mat, dtype)
        return csr.to_csc()

    else:
        raise TypeError(
            f"Cannot convert {type(mat).__name__} to SclCSC. "
            f"Supported types: SclCSR, SclCSC, scipy.sparse, numpy.ndarray"
        )


def ensure_vector(
    vec: VectorInput,
    size: Optional[int] = None,
    copy: bool = False,
) -> "RealArray":
    """Convert any vector input to RealArray.

    Args:
        vec: Input vector in any supported format.
        size: Expected size (for validation).
        copy: If True, always create a copy.

    Returns:
        RealArray containing the vector data.

    Raises:
        ValueError: If size doesn't match expected.
    """
    from scl.array import RealArray

    if isinstance(vec, RealArray):
        if copy:
            result = vec.copy()
        else:
            result = vec
    elif is_numpy_array(vec):
        result = RealArray.from_sequence(vec.ravel().tolist())
    else:
        result = RealArray.from_sequence(list(vec))

    if size is not None and result.size != size:
        raise ValueError(f"Vector size {result.size} != expected {size}")

    return result


def ensure_index_vector(
    vec: IndexInput,
    size: Optional[int] = None,
    copy: bool = False,
) -> "IndexArray":
    """Convert any index input to IndexArray.

    Args:
        vec: Input index vector.
        size: Expected size (for validation).
        copy: If True, always create a copy.

    Returns:
        IndexArray containing the indices.
    """
    from scl.array import IndexArray

    if isinstance(vec, IndexArray):
        if copy:
            result = vec.copy()
        else:
            result = vec
    elif is_numpy_array(vec):
        result = IndexArray.from_sequence(vec.ravel().astype(int).tolist())
    else:
        result = IndexArray.from_sequence([int(x) for x in vec])

    if size is not None and result.size != size:
        raise ValueError(f"Index vector size {result.size} != expected {size}")

    return result


def ensure_mask(
    mask: MaskInput,
    size: Optional[int] = None,
) -> "ByteArray":
    """Convert any mask input to ByteArray.

    Args:
        mask: Input boolean mask.
        size: Expected size (for validation).

    Returns:
        ByteArray containing the mask (0 or 1 values).
    """
    from scl.array import ByteArray

    if isinstance(mask, ByteArray):
        result = mask
    elif is_numpy_array(mask):
        result = ByteArray.from_mask(mask.ravel().tolist())
    else:
        result = ByteArray.from_mask(list(mask))

    if size is not None and result.size != size:
        raise ValueError(f"Mask size {result.size} != expected {size}")

    return result


# =============================================================================
# Internal Helpers
# =============================================================================

def _dense_to_csr(
    dense: Union["np.ndarray", Sequence[Sequence[float]]],
    dtype: str = "float64",
) -> "SclCSR":
    """Convert dense matrix to SclCSR.

    Args:
        dense: 2D dense array or nested sequence.
        dtype: Target dtype.

    Returns:
        SclCSR sparse matrix.
    """
    from scl.sparse import SclCSR
    from scl.array import RealArray, IndexArray

    # Convert to list of lists if needed
    if is_numpy_array(dense):
        rows_list = dense.tolist()
    else:
        rows_list = [list(row) for row in dense]

    n_rows = len(rows_list)
    n_cols = len(rows_list[0]) if n_rows > 0 else 0

    # Build CSR arrays
    data_list = []
    indices_list = []
    indptr_list = [0]

    for row in rows_list:
        for j, val in enumerate(row):
            if val != 0:
                data_list.append(float(val))
                indices_list.append(j)
        indptr_list.append(len(data_list))

    return SclCSR.from_arrays(
        data_list, indices_list, indptr_list, (n_rows, n_cols)
    )


def _to_numpy_if_needed(arr: Any) -> Any:
    """Convert SCL array to numpy if numpy is available.

    Args:
        arr: Input array (RealArray, IndexArray, or numpy).

    Returns:
        numpy array if numpy available, else original.
    """
    try:
        import numpy as np
        from scl.array import RealArray, IndexArray, ByteArray, to_numpy

        if isinstance(arr, (RealArray, IndexArray, ByteArray)):
            return to_numpy(arr)
        return arr
    except ImportError:
        return arr


# =============================================================================
# Decorator for Multi-Format Support
# =============================================================================

def dispatch_sparse(
    csr_func: Optional[Callable] = None,
    csc_func: Optional[Callable] = None,
    prefer_format: str = "csr",
):
    """Decorator for creating multi-format sparse functions.

    This decorator allows defining functions that work with multiple
    sparse matrix formats by dispatching to format-specific implementations.

    Args:
        csr_func: Function for CSR input (optional).
        csc_func: Function for CSC input (optional).
        prefer_format: Preferred format when input is neither CSR nor CSC.

    Returns:
        Decorated function that auto-dispatches based on input format.

    Example:
        >>> @dispatch_sparse(prefer_format="csr")
        ... def my_operation(mat):
        ...     # Implementation for CSR
        ...     return result
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(mat: SparseInput, *args, **kwargs):
            fmt = get_format(mat)

            # CSR path
            if fmt in ("scl_csr", "scipy_csr"):
                scl_mat = ensure_scl_csr(mat, copy=False)
                return func(scl_mat, *args, **kwargs)

            # CSC path
            elif fmt in ("scl_csc", "scipy_csc"):
                scl_mat = ensure_scl_csc(mat, copy=False)
                if csc_func is not None:
                    return csc_func(scl_mat, *args, **kwargs)
                # Convert to CSR if no CSC implementation
                return func(scl_mat.to_csr(), *args, **kwargs)

            # Other formats - convert to preferred
            else:
                if prefer_format == "csc":
                    scl_mat = ensure_scl_csc(mat, copy=False)
                    if csc_func is not None:
                        return csc_func(scl_mat, *args, **kwargs)
                    return func(scl_mat.to_csr(), *args, **kwargs)
                else:
                    scl_mat = ensure_scl_csr(mat, copy=False)
                    return func(scl_mat, *args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Type variables
    "T",
    "ArrayT",
    "SparseT",
    # Protocols
    "ArrayLike",
    "SparseLike",
    "CSRLike",
    "CSCLike",
    # Type aliases
    "DenseInput",
    "SparseInput",
    "CSRInput",
    "CSCInput",
    "VectorInput",
    "IndexInput",
    "MaskInput",
    "ArrayOutput",
    "IndexOutput",
    # Detection functions
    "is_scl_csr",
    "is_scl_csc",
    "is_scipy_csr",
    "is_scipy_csc",
    "is_scipy_sparse",
    "is_numpy_array",
    "get_format",
    # Conversion functions
    "ensure_scl_csr",
    "ensure_scl_csc",
    "ensure_vector",
    "ensure_index_vector",
    "ensure_mask",
    # Decorators
    "dispatch_sparse",
]
