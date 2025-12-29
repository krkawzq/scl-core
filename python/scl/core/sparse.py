"""
Sparse Matrix - Safe sparse matrix wrappers with Registry-based lifecycle management.

Provides:
    - SparseMatrixBase: Abstract base class with common operations
    - CsrMatrix: Compressed Sparse Row format (row-oriented operations)
    - CscMatrix: Compressed Sparse Column format (column-oriented operations)

Ownership Semantics:
    - copy(): Deep copy - SCL Registry fully owns the data
    - wrap(): Zero-copy view - Python holds NumPy refs, SCL holds handle
    - wrap_and_own(): Zero-copy - Ownership transferred to SCL Registry

Library Selection:
    - Automatic: Library is inferred from data dtypes
    - No need to specify lib parameter in high-level APIs
    - Use scl.set_precision() to change global defaults
"""

from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING, Optional, Tuple, Union, List, TypeVar, Any

import numpy as np

from .error import check_error
from .config import _infer_library, get_default_library

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix, csc_matrix
    from .._bindings._loader import Library


# =============================================================================
# Lazy Imports (avoid strong scipy dependency)
# =============================================================================

def _import_scipy_sparse():
    """Lazy import of scipy.sparse (only when needed)."""
    try:
        import scipy.sparse
        return scipy.sparse
    except ImportError as e:
        raise ImportError(
            "scipy is required for SciPy integration. "
            "Install with: pip install scipy"
        ) from e


def _check_scipy_sparse(obj: Any) -> bool:
    """Check if object is scipy sparse matrix (without importing scipy)."""
    # Check by module name to avoid import
    module = type(obj).__module__
    return module is not None and module.startswith('scipy.sparse')


# Type variable for self-referencing return types
T = TypeVar('T', bound='SparseMatrixBase')


class BlockStrategy(IntEnum):
    """
    Block allocation strategy for sparse matrices.
    
    Matches C API scl_block_strategy_t.
    """
    CONTIGUOUS = 0  # Single contiguous block (traditional CSR/CSC)
    SMALL = 1       # Small blocks (1K-16K elements)
    LARGE = 2       # Large blocks (64K-1M elements)
    ADAPTIVE = 3    # Auto-tune based on matrix properties


class OwnershipMode(IntEnum):
    """Ownership mode for sparse matrix data."""
    OWNED = 0       # SCL Registry owns data (from copy/create)
    WRAPPED = 1     # Zero-copy view (Python holds refs)
    TRANSFERRED = 2 # Ownership transferred to SCL (wrap_and_own)


# =============================================================================
# Base Class
# =============================================================================

class SparseMatrixBase(ABC):
    """
    Abstract base class for sparse matrices.
    
    Provides common functionality for both CSR and CSC formats.
    Use CsrMatrix or CscMatrix for specific format operations.
    """
    
    __slots__ = ("_handle", "_lib", "_ownership", "_data_refs")
    
    def __init__(
        self,
        handle: ctypes.c_void_p,
        lib: "Library",
        ownership: OwnershipMode = OwnershipMode.OWNED,
        data_refs: Optional[List] = None,
    ):
        """Internal constructor - use factory methods instead."""
        self._handle = handle
        self._lib = lib
        self._ownership = ownership
        self._data_refs = data_refs if data_refs is not None else []
    
    def __del__(self):
        self.destroy()
    
    def __enter__(self: T) -> T:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()
    
    def destroy(self) -> None:
        """
        Release native resources.
        
        Order: First C++ handle, then Python refs.
        """
        if self._handle is not None:
            handle_ptr = ctypes.pointer(self._handle)
            self._lib.scl_sparse_destroy(handle_ptr)
            self._handle = None
        self._data_refs.clear()
    
    @property
    def handle(self) -> ctypes.c_void_p:
        """Get the native handle."""
        if self._handle is None:
            raise ValueError("Matrix has been destroyed")
        return self._handle
    
    @property
    def ownership(self) -> OwnershipMode:
        """Get ownership mode."""
        return self._ownership
    
    @property
    def is_owned(self) -> bool:
        """Check if SCL owns the data."""
        return self._ownership in (OwnershipMode.OWNED, OwnershipMode.TRANSFERRED)
    
    @property
    def is_view(self) -> bool:
        """Check if this is a zero-copy view."""
        return self._ownership == OwnershipMode.WRAPPED
    
    # =========================================================================
    # Common Properties
    # =========================================================================
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get matrix shape (rows, cols)."""
        rows = self._lib.index_type()
        cols = self._lib.index_type()
        check_error(self._lib.scl_sparse_rows(self.handle, ctypes.byref(rows)), "get rows")
        check_error(self._lib.scl_sparse_cols(self.handle, ctypes.byref(cols)), "get cols")
        return (rows.value, cols.value)
    
    @property
    def nnz(self) -> int:
        """Get number of non-zero elements."""
        out = self._lib.index_type()
        check_error(self._lib.scl_sparse_nnz(self.handle, ctypes.byref(out)), "get nnz")
        return out.value
    
    @property
    def nrows(self) -> int:
        return self.shape[0]
    
    @property
    def ncols(self) -> int:
        return self.shape[1]
    
    @property
    @abstractmethod
    def format(self) -> str:
        """Get matrix format ('csr' or 'csc')."""
        pass
    
    @property
    def dtype(self) -> np.dtype:
        """Get data type (matches library variant)."""
        if self._lib.real_type == ctypes.c_float:
            return np.dtype(np.float32)
        return np.dtype(np.float64)
    
    @property
    def index_dtype(self) -> np.dtype:
        """Get index type (matches library variant)."""
        if self._lib.index_type == ctypes.c_int32:
            return np.dtype(np.int32)
        return np.dtype(np.int64)
    
    @property
    def is_contiguous(self) -> bool:
        """Check if matrix uses contiguous storage."""
        out = ctypes.c_int()
        check_error(self._lib.scl_sparse_is_contiguous(self.handle, ctypes.byref(out)), "is_contiguous")
        return bool(out.value)
    
    # =========================================================================
    # Common Operations
    # =========================================================================
    
    @abstractmethod
    def clone(self: T) -> T:
        """Create a deep copy of this matrix."""
        pass
    
    def to_contiguous(self: T) -> T:
        """Convert to contiguous storage format."""
        handle = ctypes.c_void_p()
        check_error(
            self._lib.scl_sparse_to_contiguous(self.handle, ctypes.byref(handle)),
            "to_contiguous",
        )
        return self.__class__(handle, self._lib, OwnershipMode.OWNED)
    
    def __len__(self) -> int:
        return self.nrows
    
    def __repr__(self) -> str:
        if self._handle is None:
            return f"<{self.__class__.__name__} [destroyed]>"
        
        rows, cols = self.shape
        ownership_str = {
            OwnershipMode.OWNED: "owned",
            OwnershipMode.WRAPPED: "view",
            OwnershipMode.TRANSFERRED: "transferred",
        }.get(self._ownership, "unknown")
        
        return f"<{self.__class__.__name__} {rows}x{cols}, nnz={self.nnz}, {ownership_str}>"


# =============================================================================
# CSR Matrix (Row-Oriented)
# =============================================================================

class CsrMatrix(SparseMatrixBase):
    """
    Compressed Sparse Row (CSR) matrix.
    
    Optimized for:
        - Row-wise operations
        - Row slicing (zero-copy views)
        - Sparse matrix-vector multiplication (A @ x)
        - Iterating over rows
    
    Examples:
        # From SciPy
        >>> mat = CsrMatrix.copy(scipy_csr_matrix)
        
        # Row slicing (zero-copy)
        >>> rows_10_20 = mat[10:20]
        
        # Transpose to CSC
        >>> csc = mat.T
    """
    
    @property
    def format(self) -> str:
        return "csr"
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def copy(
        cls,
        source: Union["csr_matrix", np.ndarray],
        strategy: BlockStrategy = BlockStrategy.ADAPTIVE,
    ) -> "CsrMatrix":
        """
        Create CSR matrix by COPYING data.
        
        Args:
            source: SciPy CSR matrix or dense NumPy array
            strategy: Block allocation strategy
            
        Returns:
            CsrMatrix with SCL-owned data
        """
        # Lazy import scipy
        sp = _import_scipy_sparse()
        
        if isinstance(source, np.ndarray):
            source = sp.csr_matrix(source)
        
        if not sp.issparse(source):
            raise TypeError(f"Expected sparse matrix or ndarray, got {type(source)}")
        
        if source.format != 'csr':
            source = source.tocsr()
        
        lib = _infer_library(data=source.data, indices=source.indices)
        return cls._create_from_arrays(
            source.indptr, source.indices, source.data,
            source.shape, lib, strategy, copy=True,
        )
    
    @classmethod
    def from_scipy(
        cls,
        matrix: "csr_matrix",
        strategy: BlockStrategy = BlockStrategy.ADAPTIVE,
    ) -> "CsrMatrix":
        """Create by copying from SciPy CSR matrix."""
        return cls.copy(matrix, strategy)
    
    @classmethod
    def wrap(
        cls,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        shape: Tuple[int, int],
    ) -> "CsrMatrix":
        """
        Create ZERO-COPY view wrapping existing arrays.
        
        WARNING: Input arrays MUST stay alive!
        """
        lib = _infer_library(data=data, indices=indices)
        return cls._create_from_arrays(
            indptr, indices, data, shape, lib,
            BlockStrategy.CONTIGUOUS, copy=False,
        )
    
    @classmethod
    def wrap_and_own(
        cls,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        shape: Tuple[int, int],
    ) -> "CsrMatrix":
        """Create zero-copy view and TRANSFER OWNERSHIP to SCL."""
        lib = _infer_library(data=data, indices=indices)
        rows, cols = shape
        nnz = len(data)
        
        real_dtype = np.float64 if lib.real_type == ctypes.c_double else np.float32
        index_dtype = np.int64 if lib.index_type == ctypes.c_int64 else np.int32
        
        if data.dtype != real_dtype:
            raise ValueError(f"Data dtype {data.dtype} != library type {real_dtype}")
        if indptr.dtype != index_dtype or indices.dtype != index_dtype:
            raise ValueError(f"Index dtype mismatch, expected {index_dtype}")
        
        if not all(arr.flags['C_CONTIGUOUS'] for arr in [data, indptr, indices]):
            raise ValueError("All arrays must be C-contiguous")
        
        handle = ctypes.c_void_p()
        check_error(
            lib.scl_sparse_wrap_and_own(
                ctypes.byref(handle), rows, cols, nnz,
                indptr.ctypes.data_as(ctypes.POINTER(lib.index_type)),
                indices.ctypes.data_as(ctypes.POINTER(lib.index_type)),
                data.ctypes.data_as(ctypes.POINTER(lib.real_type)),
                1,  # is_csr = True
            ),
            "wrap_and_own",
        )
        return cls(handle, lib, OwnershipMode.TRANSFERRED)
    
    @classmethod
    def from_coo(
        cls,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
        values: np.ndarray,
        shape: Tuple[int, int],
        strategy: BlockStrategy = BlockStrategy.ADAPTIVE,
    ) -> "CsrMatrix":
        """Create from COO format."""
        lib = _infer_library(data=values, indices=row_indices)
        rows, cols = shape
        nnz = len(values)
        
        real_dtype = np.float64 if lib.real_type == ctypes.c_double else np.float32
        index_dtype = np.int64 if lib.index_type == ctypes.c_int64 else np.int32
        
        row_indices = np.ascontiguousarray(row_indices, dtype=index_dtype)
        col_indices = np.ascontiguousarray(col_indices, dtype=index_dtype)
        values = np.ascontiguousarray(values, dtype=real_dtype)
        
        handle = ctypes.c_void_p()
        check_error(
            lib.scl_sparse_from_coo(
                ctypes.byref(handle), rows, cols, nnz,
                row_indices.ctypes.data_as(ctypes.POINTER(lib.index_type)),
                col_indices.ctypes.data_as(ctypes.POINTER(lib.index_type)),
                values.ctypes.data_as(ctypes.POINTER(lib.real_type)),
                1, int(strategy),  # is_csr = True
            ),
            "from_coo",
        )
        return cls(handle, lib, OwnershipMode.OWNED)
    
    @classmethod
    def _create_from_arrays(
        cls,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        shape: Tuple[int, int],
        lib: "Library",
        strategy: BlockStrategy,
        copy: bool,
    ) -> "CsrMatrix":
        """Internal helper."""
        rows, cols = shape
        nnz = len(data)
        
        real_dtype = np.float64 if lib.real_type == ctypes.c_double else np.float32
        index_dtype = np.int64 if lib.index_type == ctypes.c_int64 else np.int32
        
        data = np.ascontiguousarray(data, dtype=real_dtype)
        indptr = np.ascontiguousarray(indptr, dtype=index_dtype)
        indices = np.ascontiguousarray(indices, dtype=index_dtype)
        
        data_ptr = data.ctypes.data_as(ctypes.POINTER(lib.real_type))
        indptr_ptr = indptr.ctypes.data_as(ctypes.POINTER(lib.index_type))
        indices_ptr = indices.ctypes.data_as(ctypes.POINTER(lib.index_type))
        
        handle = ctypes.c_void_p()
        
        if copy:
            check_error(
                lib.scl_sparse_create_with_strategy(
                    ctypes.byref(handle), rows, cols, nnz,
                    indptr_ptr, indices_ptr, data_ptr,
                    1, int(strategy),  # is_csr = True
                ),
                "create CSR matrix",
            )
            return cls(handle, lib, OwnershipMode.OWNED)
        else:
            check_error(
                lib.scl_sparse_wrap(
                    ctypes.byref(handle), rows, cols, nnz,
                    indptr_ptr, indices_ptr, data_ptr, 1,
                ),
                "wrap CSR matrix",
            )
            return cls(handle, lib, OwnershipMode.WRAPPED, data_refs=[data, indptr, indices])
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def to_scipy(self) -> "csr_matrix":
        """Export to SciPy CSR matrix."""
        # Lazy import scipy
        sp = _import_scipy_sparse()
        
        rows, cols = self.shape
        nnz = self.nnz
        
        indptr = np.empty(rows + 1, dtype=self.index_dtype)
        indices = np.empty(nnz, dtype=self.index_dtype)
        data = np.empty(nnz, dtype=self.dtype)
        
        check_error(
            self._lib.scl_sparse_export(
                self.handle,
                indptr.ctypes.data_as(ctypes.POINTER(self._lib.index_type)),
                indices.ctypes.data_as(ctypes.POINTER(self._lib.index_type)),
                data.ctypes.data_as(ctypes.POINTER(self._lib.real_type)),
            ),
            "export",
        )
        return sp.csr_matrix((data, indices, indptr), shape=(rows, cols))
    
    def to_dense(self) -> np.ndarray:
        """Convert to dense NumPy array."""
        return self.to_scipy().toarray()
    
    # =========================================================================
    # CSR-Specific Operations
    # =========================================================================
    
    def clone(self) -> "CsrMatrix":
        """Create a deep copy."""
        handle = ctypes.c_void_p()
        check_error(self._lib.scl_sparse_clone(self.handle, ctypes.byref(handle)), "clone")
        return CsrMatrix(handle, self._lib, OwnershipMode.OWNED)
    
    def transpose(self) -> "CscMatrix":
        """
        Transpose to CSC format.
        
        Returns:
            CscMatrix (transposed)
        """
        handle = ctypes.c_void_p()
        check_error(self._lib.scl_sparse_transpose(self.handle, ctypes.byref(handle)), "transpose")
        return CscMatrix(handle, self._lib, OwnershipMode.OWNED)
    
    @property
    def T(self) -> "CscMatrix":
        """Transpose shorthand."""
        return self.transpose()
    
    # =========================================================================
    # Row Slicing (CSR specialty)
    # =========================================================================
    
    def row_slice_view(self, start: int, end: int) -> "CsrMatrix":
        """
        Get row range view (zero-copy).
        
        Args:
            start: Start row index [0, rows)
            end: End row index (start, rows]
            
        Returns:
            CsrMatrix view sharing data with parent
        """
        handle = ctypes.c_void_p()
        check_error(
            self._lib.scl_sparse_row_range_view(self.handle, start, end, ctypes.byref(handle)),
            "row_slice_view",
        )
        mat = CsrMatrix(handle, self._lib, OwnershipMode.WRAPPED)
        mat._data_refs = [self]
        return mat
    
    def row_slice_copy(
        self,
        start: int,
        end: int,
        strategy: BlockStrategy = BlockStrategy.ADAPTIVE,
    ) -> "CsrMatrix":
        """Get row range with copy."""
        handle = ctypes.c_void_p()
        check_error(
            self._lib.scl_sparse_row_range_copy(self.handle, start, end, int(strategy), ctypes.byref(handle)),
            "row_slice_copy",
        )
        return CsrMatrix(handle, self._lib, OwnershipMode.OWNED)
    
    def select_rows(
        self,
        indices: np.ndarray,
        copy: bool = False,
        strategy: BlockStrategy = BlockStrategy.ADAPTIVE,
    ) -> "CsrMatrix":
        """
        Select rows by indices.
        
        Args:
            indices: Row indices to select
            copy: If True, copy data; if False, create view
            strategy: Block strategy for copy
        """
        indices = np.ascontiguousarray(indices, dtype=self.index_dtype)
        indices_ptr = indices.ctypes.data_as(ctypes.POINTER(self._lib.index_type))
        
        handle = ctypes.c_void_p()
        
        if copy:
            check_error(
                self._lib.scl_sparse_row_slice_copy(
                    self.handle, indices_ptr, len(indices), int(strategy), ctypes.byref(handle),
                ),
                "select_rows (copy)",
            )
            return CsrMatrix(handle, self._lib, OwnershipMode.OWNED)
        else:
            check_error(
                self._lib.scl_sparse_slice_rows(self.handle, indices_ptr, len(indices), ctypes.byref(handle)),
                "select_rows",
            )
            mat = CsrMatrix(handle, self._lib, OwnershipMode.WRAPPED)
            mat._data_refs = [self]
            return mat
    
    def __getitem__(self, key) -> "CsrMatrix":
        """
        NumPy-style row indexing.
        
        Supports:
            mat[5:10]     - Row slice (view)
            mat[[1,3,5]]  - Row selection
            mat[5]        - Single row
        """
        if isinstance(key, slice):
            start, stop, step = key.indices(self.nrows)
            if step == 1:
                return self.row_slice_view(start, stop)
            indices = np.arange(start, stop, step, dtype=self.index_dtype)
            return self.select_rows(indices)
        
        if isinstance(key, (list, np.ndarray)):
            return self.select_rows(np.asarray(key))
        
        if isinstance(key, int):
            if key < 0:
                key += self.nrows
            return self.row_slice_view(key, key + 1)
        
        raise TypeError(f"Unsupported index type: {type(key)}")
    
    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    @classmethod
    def vstack(
        cls,
        matrices: List["CsrMatrix"],
        strategy: BlockStrategy = BlockStrategy.ADAPTIVE,
    ) -> "CsrMatrix":
        """
        Vertical stack (concatenate by rows).
        
        All matrices must have same number of columns.
        """
        if not matrices:
            raise ValueError("Cannot vstack empty list")
        
        lib = matrices[0]._lib
        n = len(matrices)
        
        handle_array = (ctypes.c_void_p * n)()
        for i, mat in enumerate(matrices):
            handle_array[i] = mat.handle
        
        out_handle = ctypes.c_void_p()
        check_error(
            lib.scl_sparse_vstack(handle_array, n, int(strategy), ctypes.byref(out_handle)),
            "vstack",
        )
        return cls(out_handle, lib, OwnershipMode.OWNED)


# =============================================================================
# CSC Matrix (Column-Oriented)
# =============================================================================

class CscMatrix(SparseMatrixBase):
    """
    Compressed Sparse Column (CSC) matrix.
    
    Optimized for:
        - Column-wise operations
        - Column slicing (zero-copy views)
        - Solving linear systems
        - Iterating over columns
    
    Examples:
        # From SciPy
        >>> mat = CscMatrix.copy(scipy_csc_matrix)
        
        # Column slicing (zero-copy)
        >>> cols_10_20 = mat[:, 10:20]
        
        # Transpose to CSR
        >>> csr = mat.T
    """
    
    @property
    def format(self) -> str:
        return "csc"
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def copy(
        cls,
        source: Union["csc_matrix", np.ndarray],
        strategy: BlockStrategy = BlockStrategy.ADAPTIVE,
    ) -> "CscMatrix":
        """
        Create CSC matrix by COPYING data.
        
        Args:
            source: SciPy CSC matrix or dense NumPy array
            strategy: Block allocation strategy
        """
        # Lazy import scipy
        sp = _import_scipy_sparse()
        
        if isinstance(source, np.ndarray):
            source = sp.csc_matrix(source)
        
        if not sp.issparse(source):
            raise TypeError(f"Expected sparse matrix or ndarray, got {type(source)}")
        
        if source.format != 'csc':
            source = source.tocsc()
        
        lib = _infer_library(data=source.data, indices=source.indices)
        return cls._create_from_arrays(
            source.indptr, source.indices, source.data,
            source.shape, lib, strategy, copy=True,
        )
    
    @classmethod
    def from_scipy(
        cls,
        matrix: "csc_matrix",
        strategy: BlockStrategy = BlockStrategy.ADAPTIVE,
    ) -> "CscMatrix":
        """Create by copying from SciPy CSC matrix."""
        return cls.copy(matrix, strategy)
    
    @classmethod
    def wrap(
        cls,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        shape: Tuple[int, int],
    ) -> "CscMatrix":
        """Create ZERO-COPY view wrapping existing arrays."""
        lib = _infer_library(data=data, indices=indices)
        return cls._create_from_arrays(
            indptr, indices, data, shape, lib,
            BlockStrategy.CONTIGUOUS, copy=False,
        )
    
    @classmethod
    def wrap_and_own(
        cls,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        shape: Tuple[int, int],
    ) -> "CscMatrix":
        """Create zero-copy view and TRANSFER OWNERSHIP to SCL."""
        lib = _infer_library(data=data, indices=indices)
        rows, cols = shape
        nnz = len(data)
        
        real_dtype = np.float64 if lib.real_type == ctypes.c_double else np.float32
        index_dtype = np.int64 if lib.index_type == ctypes.c_int64 else np.int32
        
        if data.dtype != real_dtype:
            raise ValueError(f"Data dtype {data.dtype} != library type {real_dtype}")
        if indptr.dtype != index_dtype or indices.dtype != index_dtype:
            raise ValueError(f"Index dtype mismatch, expected {index_dtype}")
        
        if not all(arr.flags['C_CONTIGUOUS'] for arr in [data, indptr, indices]):
            raise ValueError("All arrays must be C-contiguous")
        
        handle = ctypes.c_void_p()
        check_error(
            lib.scl_sparse_wrap_and_own(
                ctypes.byref(handle), rows, cols, nnz,
                indptr.ctypes.data_as(ctypes.POINTER(lib.index_type)),
                indices.ctypes.data_as(ctypes.POINTER(lib.index_type)),
                data.ctypes.data_as(ctypes.POINTER(lib.real_type)),
                0,  # is_csr = False
            ),
            "wrap_and_own",
        )
        return cls(handle, lib, OwnershipMode.TRANSFERRED)
    
    @classmethod
    def from_coo(
        cls,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
        values: np.ndarray,
        shape: Tuple[int, int],
        strategy: BlockStrategy = BlockStrategy.ADAPTIVE,
    ) -> "CscMatrix":
        """Create from COO format."""
        lib = _infer_library(data=values, indices=row_indices)
        rows, cols = shape
        nnz = len(values)
        
        real_dtype = np.float64 if lib.real_type == ctypes.c_double else np.float32
        index_dtype = np.int64 if lib.index_type == ctypes.c_int64 else np.int32
        
        row_indices = np.ascontiguousarray(row_indices, dtype=index_dtype)
        col_indices = np.ascontiguousarray(col_indices, dtype=index_dtype)
        values = np.ascontiguousarray(values, dtype=real_dtype)
        
        handle = ctypes.c_void_p()
        check_error(
            lib.scl_sparse_from_coo(
                ctypes.byref(handle), rows, cols, nnz,
                row_indices.ctypes.data_as(ctypes.POINTER(lib.index_type)),
                col_indices.ctypes.data_as(ctypes.POINTER(lib.index_type)),
                values.ctypes.data_as(ctypes.POINTER(lib.real_type)),
                0, int(strategy),  # is_csr = False
            ),
            "from_coo",
        )
        return cls(handle, lib, OwnershipMode.OWNED)
    
    @classmethod
    def _create_from_arrays(
        cls,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        shape: Tuple[int, int],
        lib: "Library",
        strategy: BlockStrategy,
        copy: bool,
    ) -> "CscMatrix":
        """Internal helper."""
        rows, cols = shape
        nnz = len(data)
        
        real_dtype = np.float64 if lib.real_type == ctypes.c_double else np.float32
        index_dtype = np.int64 if lib.index_type == ctypes.c_int64 else np.int32
        
        data = np.ascontiguousarray(data, dtype=real_dtype)
        indptr = np.ascontiguousarray(indptr, dtype=index_dtype)
        indices = np.ascontiguousarray(indices, dtype=index_dtype)
        
        data_ptr = data.ctypes.data_as(ctypes.POINTER(lib.real_type))
        indptr_ptr = indptr.ctypes.data_as(ctypes.POINTER(lib.index_type))
        indices_ptr = indices.ctypes.data_as(ctypes.POINTER(lib.index_type))
        
        handle = ctypes.c_void_p()
        
        if copy:
            check_error(
                lib.scl_sparse_create_with_strategy(
                    ctypes.byref(handle), rows, cols, nnz,
                    indptr_ptr, indices_ptr, data_ptr,
                    0, int(strategy),  # is_csr = False
                ),
                "create CSC matrix",
            )
            return cls(handle, lib, OwnershipMode.OWNED)
        else:
            check_error(
                lib.scl_sparse_wrap(
                    ctypes.byref(handle), rows, cols, nnz,
                    indptr_ptr, indices_ptr, data_ptr, 0,
                ),
                "wrap CSC matrix",
            )
            return cls(handle, lib, OwnershipMode.WRAPPED, data_refs=[data, indptr, indices])
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def to_scipy(self) -> "csc_matrix":
        """Export to SciPy CSC matrix."""
        # Lazy import scipy
        sp = _import_scipy_sparse()
        
        rows, cols = self.shape
        nnz = self.nnz
        
        indptr = np.empty(cols + 1, dtype=self.index_dtype)
        indices = np.empty(nnz, dtype=self.index_dtype)
        data = np.empty(nnz, dtype=self.dtype)
        
        check_error(
            self._lib.scl_sparse_export(
                self.handle,
                indptr.ctypes.data_as(ctypes.POINTER(self._lib.index_type)),
                indices.ctypes.data_as(ctypes.POINTER(self._lib.index_type)),
                data.ctypes.data_as(ctypes.POINTER(self._lib.real_type)),
            ),
            "export",
        )
        return sp.csc_matrix((data, indices, indptr), shape=(rows, cols))
    
    def to_dense(self) -> np.ndarray:
        """Convert to dense NumPy array."""
        return self.to_scipy().toarray()
    
    # =========================================================================
    # CSC-Specific Operations
    # =========================================================================
    
    def clone(self) -> "CscMatrix":
        """Create a deep copy."""
        handle = ctypes.c_void_p()
        check_error(self._lib.scl_sparse_clone(self.handle, ctypes.byref(handle)), "clone")
        return CscMatrix(handle, self._lib, OwnershipMode.OWNED)
    
    def transpose(self) -> "CsrMatrix":
        """
        Transpose to CSR format.
        
        Returns:
            CsrMatrix (transposed)
        """
        handle = ctypes.c_void_p()
        check_error(self._lib.scl_sparse_transpose(self.handle, ctypes.byref(handle)), "transpose")
        return CsrMatrix(handle, self._lib, OwnershipMode.OWNED)
    
    @property
    def T(self) -> "CsrMatrix":
        """Transpose shorthand."""
        return self.transpose()
    
    # =========================================================================
    # Column Slicing (CSC specialty)
    # =========================================================================
    
    def col_slice_view(self, start: int, end: int) -> "CscMatrix":
        """
        Get column range view (zero-copy).
        
        Note: CSC uses scl_sparse_slice_cols for column views.
        """
        # For CSC, column slice is like row slice for CSR
        indices = np.arange(start, end, dtype=self.index_dtype)
        return self.select_cols(indices)
    
    def select_cols(
        self,
        indices: np.ndarray,
        copy: bool = False,
        strategy: BlockStrategy = BlockStrategy.ADAPTIVE,
    ) -> "CscMatrix":
        """
        Select columns by indices.
        
        Args:
            indices: Column indices to select
            copy: If True, copy data; if False, create view
            strategy: Block strategy for copy
        """
        indices = np.ascontiguousarray(indices, dtype=self.index_dtype)
        indices_ptr = indices.ctypes.data_as(ctypes.POINTER(self._lib.index_type))
        
        handle = ctypes.c_void_p()
        
        if copy:
            check_error(
                self._lib.scl_sparse_col_slice(
                    self.handle, indices_ptr, len(indices), int(strategy), ctypes.byref(handle),
                ),
                "select_cols (copy)",
            )
            return CscMatrix(handle, self._lib, OwnershipMode.OWNED)
        else:
            check_error(
                self._lib.scl_sparse_slice_cols(self.handle, indices_ptr, len(indices), ctypes.byref(handle)),
                "select_cols",
            )
            mat = CscMatrix(handle, self._lib, OwnershipMode.WRAPPED)
            mat._data_refs = [self]
            return mat
    
    def __getitem__(self, key) -> "CscMatrix":
        """
        NumPy-style column indexing.
        
        Supports:
            mat[5:10]     - Column slice (view)
            mat[[1,3,5]]  - Column selection
            mat[5]        - Single column
        """
        if isinstance(key, slice):
            start, stop, step = key.indices(self.ncols)
            if step == 1:
                return self.col_slice_view(start, stop)
            indices = np.arange(start, stop, step, dtype=self.index_dtype)
            return self.select_cols(indices)
        
        if isinstance(key, (list, np.ndarray)):
            return self.select_cols(np.asarray(key))
        
        if isinstance(key, int):
            if key < 0:
                key += self.ncols
            return self.col_slice_view(key, key + 1)
        
        raise TypeError(f"Unsupported index type: {type(key)}")
    
    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    @classmethod
    def hstack(
        cls,
        matrices: List["CscMatrix"],
        strategy: BlockStrategy = BlockStrategy.ADAPTIVE,
    ) -> "CscMatrix":
        """
        Horizontal stack (concatenate by columns).
        
        All matrices must have same number of rows.
        """
        if not matrices:
            raise ValueError("Cannot hstack empty list")
        
        lib = matrices[0]._lib
        n = len(matrices)
        
        handle_array = (ctypes.c_void_p * n)()
        for i, mat in enumerate(matrices):
            handle_array[i] = mat.handle
        
        out_handle = ctypes.c_void_p()
        check_error(
            lib.scl_sparse_hstack(handle_array, n, int(strategy), ctypes.byref(out_handle)),
            "hstack",
        )
        return cls(out_handle, lib, OwnershipMode.OWNED)


# =============================================================================
# Convenience Alias
# =============================================================================

# Default sparse matrix type (most common use case)
SparseMatrix = CsrMatrix
