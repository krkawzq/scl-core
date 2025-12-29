"""
DenseMatrix - Unsafe dense matrix wrapper.

WARNING: This is an UNSAFE interface by design!

DenseMatrix is a lightweight wrapper around external memory. It does NOT:
- Own any data
- Hold references to source data
- Guarantee data validity or safety

The caller is FULLY RESPONSIBLE for:
- Ensuring data stays alive during matrix lifetime
- Managing data memory lifecycle
- Thread synchronization for concurrent access

This design is intentional - it provides maximum flexibility and zero overhead
for advanced users who need direct memory access. For safe operations, use
SparseMatrix which has proper lifecycle management via Registry.

Typical usage:
    # User manages numpy array lifetime
    arr = np.random.randn(1000, 100)
    
    with DenseMatrix.wrap(arr) as mat:
        # arr MUST stay alive here!
        result = some_operation(sparse_mat, mat)
    # mat handle released, arr can now be freed
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Tuple

import numpy as np

from .error import check_error
from .config import _infer_library, get_default_library

if TYPE_CHECKING:
    from .._bindings._loader import Library


class DenseMatrix:
    """
    Unsafe dense matrix wrapper for SCL.
    
    This is a thin handle wrapper with NO safety guarantees.
    The underlying data is NOT owned or referenced by this class.
    
    Library selection is AUTOMATIC based on data dtype.
    
    WARNING:
        - Data validity is NOT guaranteed
        - Caller MUST ensure data survives matrix lifetime
        - No automatic memory management
        - Not suitable for long-lived objects
    
    Use this only for:
        - Temporary views during computation
        - Advanced users who understand memory management
        - Performance-critical code where overhead matters
    
    For safe matrix operations, use SparseMatrix instead.
    """
    
    __slots__ = ("_handle", "_lib")
    
    def __init__(self, handle: ctypes.c_void_p, lib: "Library"):
        """
        Initialize from native handle.
        
        Internal constructor - use wrap() factory method instead.
        
        Args:
            handle: Native SCL dense matrix handle
            lib: Library instance (internal, auto-selected)
        """
        self._handle = handle
        self._lib = lib
    
    def __del__(self):
        """Release handle (NOT data - we never own it)."""
        self.destroy()
    
    def __enter__(self) -> "DenseMatrix":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()
    
    def destroy(self) -> None:
        """
        Release the native handle.
        
        NOTE: This does NOT free any data - DenseMatrix never owns data.
        The underlying memory must be managed by the caller.
        """
        if self._handle is not None:
            handle_ptr = ctypes.pointer(self._handle)
            self._lib.scl_dense_destroy(handle_ptr)
            self._handle = None
    
    @property
    def handle(self) -> ctypes.c_void_p:
        """
        Get the native handle.
        
        WARNING: The handle points to external data that may be invalid!
        Caller must ensure data is still alive.
        """
        if self._handle is None:
            raise ValueError("Matrix handle has been destroyed")
        return self._handle
    
    # =========================================================================
    # Properties (All unsafe - data may be invalid)
    # =========================================================================
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get matrix shape (rows, cols). Data validity NOT guaranteed."""
        rows = self._lib.index_type()
        cols = self._lib.index_type()
        check_error(self._lib.scl_dense_rows(self.handle, ctypes.byref(rows)), "get rows")
        check_error(self._lib.scl_dense_cols(self.handle, ctypes.byref(cols)), "get cols")
        return (rows.value, cols.value)
    
    @property
    def nrows(self) -> int:
        """Get number of rows."""
        return self.shape[0]
    
    @property
    def ncols(self) -> int:
        """Get number of columns."""
        return self.shape[1]
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        out = ctypes.c_size_t()
        check_error(self._lib.scl_dense_size(self.handle, ctypes.byref(out)), "get size")
        return out.value
    
    @property
    def stride(self) -> int:
        """Get row stride."""
        out = self._lib.index_type()
        check_error(self._lib.scl_dense_stride(self.handle, ctypes.byref(out)), "get stride")
        return out.value
    
    @property
    def is_contiguous(self) -> bool:
        """Check if matrix is contiguous (stride == cols)."""
        out = ctypes.c_int()
        check_error(self._lib.scl_dense_is_contiguous(self.handle, ctypes.byref(out)), "is_contiguous")
        return bool(out.value)
    
    @property
    def dtype(self) -> np.dtype:
        """Get data type (matches library variant)."""
        real_type = self._lib.real_type
        if real_type == ctypes.c_float:
            return np.dtype(np.float32)
        return np.dtype(np.float64)
    
    # =========================================================================
    # Factory Method
    # =========================================================================
    
    @classmethod
    def wrap(cls, data: np.ndarray) -> "DenseMatrix":
        """
        Create UNSAFE view wrapping external data.
        
        Library is automatically selected based on data dtype.
        
        WARNING: This creates an UNSAFE view!
        - No data is copied
        - No reference to data is held
        - Data validity is NOT guaranteed
        - Caller MUST keep data alive during matrix lifetime
        
        Args:
            data: NumPy array (1D or 2D, must be C-contiguous)
            
        Returns:
            DenseMatrix view (UNSAFE - does not own or reference data)
            
        Raises:
            ValueError: If data is not compatible
            
        Example:
            # CORRECT: data stays alive
            arr = np.random.randn(100, 50)
            mat = DenseMatrix.wrap(arr)
            result = some_operation(sparse, mat)
            mat.destroy()  # Safe now
            
            # WRONG: data may be freed!
            mat = DenseMatrix.wrap(np.random.randn(100, 50))
            # The numpy array may be garbage collected!
            result = some_operation(sparse, mat)  # UNDEFINED BEHAVIOR
        """
        # Auto-select library based on data dtype
        lib = _infer_library(data=data)
        
        # Handle 1D arrays as (n, 1) column vectors
        if data.ndim == 1:
            rows = len(data)
            cols = 1
            stride = 1
        elif data.ndim == 2:
            rows, cols = data.shape
            stride = data.strides[0] // data.itemsize
        else:
            raise ValueError(f"Expected 1D or 2D array, got {data.ndim}D")
        
        # Check dtype compatibility
        real_dtype = np.float64 if lib.real_type == ctypes.c_double else np.float32
        if data.dtype != real_dtype:
            raise ValueError(
                f"Data dtype {data.dtype} does not match library type {real_dtype}. "
                f"Convert data before wrapping (no implicit copy for safety)."
            )
        
        # Check contiguity
        if not data.flags['C_CONTIGUOUS']:
            raise ValueError(
                "Data must be C-contiguous. "
                "Use np.ascontiguousarray() before wrapping."
            )
        
        data_ptr = data.ctypes.data_as(ctypes.POINTER(lib.real_type))
        
        handle = ctypes.c_void_p()
        check_error(
            lib.scl_dense_wrap(
                ctypes.byref(handle),
                rows, cols, data_ptr, stride,
            ),
            "wrap dense matrix",
        )
        
        return cls(handle, lib)
    
    # =========================================================================
    # Element Access (All unsafe - data may be invalid)
    # =========================================================================
    
    def get(self, row: int, col: int) -> float:
        """
        Get element at (row, col).
        
        WARNING: Data validity NOT guaranteed!
        """
        out = self._lib.real_type()
        check_error(
            self._lib.scl_dense_get(self.handle, row, col, ctypes.byref(out)),
            "get element",
        )
        return out.value
    
    def set(self, row: int, col: int, value: float) -> None:
        """
        Set element at (row, col).
        
        WARNING: Data validity NOT guaranteed!
        """
        check_error(
            self._lib.scl_dense_set(self.handle, row, col, value),
            "set element",
        )
    
    def fill(self, value: float) -> None:
        """
        Fill matrix with scalar value (in-place).
        
        WARNING: Data validity NOT guaranteed!
        """
        check_error(
            self._lib.scl_dense_fill(self.handle, value),
            "fill matrix",
        )
    
    def __getitem__(self, key) -> float:
        """Get element at (row, col)."""
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Index must be (row, col) tuple")
        return self.get(key[0], key[1])
    
    def __setitem__(self, key, value: float) -> None:
        """Set element at (row, col)."""
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Index must be (row, col) tuple")
        self.set(key[0], key[1], value)
    
    # =========================================================================
    # Magic Methods
    # =========================================================================
    
    def __repr__(self) -> str:
        if self._handle is None:
            return "<DenseMatrix [destroyed]>"
        rows, cols = self.shape
        return f"<DenseMatrix {rows}x{cols} [UNSAFE - data may be invalid]>"
    
    def __len__(self) -> int:
        return self.nrows


# =============================================================================
# Array Helper (for 1D data)
# =============================================================================

class ArrayView(DenseMatrix):
    """
    UNSAFE 1D array view.
    
    Convenience wrapper for 1D NumPy arrays, treated as column vectors.
    Same safety warnings as DenseMatrix apply.
    """
    
    @classmethod
    def wrap(cls, data: np.ndarray) -> "ArrayView":
        """
        Create UNSAFE view of 1D array.
        
        Args:
            data: 1D NumPy array (must be contiguous)
            
        Returns:
            ArrayView (UNSAFE - does not own or reference data)
        """
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array, got {data.ndim}D")
        
        # Use parent implementation
        mat = DenseMatrix.wrap(data)
        # Re-wrap as ArrayView
        return cls(mat._handle, mat._lib)
    
    @property
    def length(self) -> int:
        """Get array length."""
        return self.nrows
    
    def __repr__(self) -> str:
        if self._handle is None:
            return "<ArrayView [destroyed]>"
        return f"<ArrayView length={self.length} [UNSAFE - data may be invalid]>"
