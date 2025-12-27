"""
Callback-Based Sparse Matrices

High-level Python classes for callback-based sparse matrices.
These allow users to implement custom data access patterns in Python
that integrate seamlessly with SCL's C++ operators.

Use Cases:

1. Lazy Loading: Load data from disk on demand
   - HDF5/Zarr files
   - Database queries
   - Network resources

2. Virtual Views: Transform data without copying
   - Filtering
   - Normalization
   - Type conversion

3. Custom Storage: Implement arbitrary storage formats
   - Compressed formats
   - Distributed storage
   - Encrypted data

Example:

    import h5py
    
    class HDF5LazyCSR(CallbackCSR):
        '''Lazy-loading CSR from HDF5 file.'''
        
        def __init__(self, filepath, dataset_name):
            self.f = h5py.File(filepath, 'r')
            self.ds = self.f[dataset_name]
            self._shape = tuple(self.ds.attrs['shape'])
            self._nnz = int(self.ds.attrs['nnz'])
            super().__init__()
        
        def get_shape(self):
            return self._shape
        
        def get_nnz(self):
            return self._nnz
        
        def get_row_data(self, i):
            # Load row i on demand
            values = self.ds[f'row_{i}/data'][:]
            indices = self.ds[f'row_{i}/indices'][:]
            return values, indices
        
        def close(self):
            self.f.close()
            super().close()
    
    # Use like any other sparse matrix
    with HDF5LazyCSR('data.h5', 'X') as mat:
        row_sums = mat.sum(axis=1)

Performance Notes:

- Each row/column access crosses the Python-C++ boundary
- Callbacks hold the GIL, preventing parallel execution
- Best suited for I/O-bound scenarios
- For compute-intensive loops, consider materializing first
"""

from typing import Tuple, Optional, Union, Any, TYPE_CHECKING
from abc import abstractmethod
import ctypes
from ctypes import POINTER, c_void_p, pointer
import numpy as np

from ._base import CSRBase, CSCBase
from ._array import Array

# Import kernel bindings
from .._kernel import callback as kernel_callback
from .._kernel.callback import (
    CallbackVTable, CallbackHandle,
    GetRowsFunc, GetColsFunc, GetNnzFunc,
    GetPrimaryValuesFunc, GetPrimaryIndicesFunc,
    GetPrimaryLengthFunc, PrefetchRangeFunc, ReleasePrimaryFunc,
)
from .._kernel.types import c_real, c_index

if TYPE_CHECKING:
    from scipy.sparse import spmatrix

__all__ = ['CallbackCSR', 'CallbackCSC']


class CallbackCSR(CSRBase):
    """
    Callback-based CSR sparse matrix.
    
    Allows Python users to define custom row data access patterns
    that integrate with SCL's C++ operators. Subclass this and
    implement the required methods.
    
    Required Methods to Implement:
        get_shape() -> Tuple[int, int]: Return (rows, cols)
        get_nnz() -> int: Return total non-zero count
        get_row_data(i) -> Tuple[ndarray, ndarray]: Return (values, indices) for row i
    
    Optional Methods to Override:
        get_row_length(i) -> int: Fast length query (default uses get_row_data)
        prefetch_rows(start, end): Batch prefetch for performance
        release_row(i): Release resources after use
        close(): Cleanup when done
    
    Memory Contract:
        Data returned by get_row_data() must remain valid until:
        - The next call to get_row_data() for the same row
        - close() is called
        - The object is garbage collected
    
    Example:
    
        class MyLazyCSR(CallbackCSR):
            def __init__(self, data_source):
                self._source = data_source
                super().__init__()
            
            def get_shape(self):
                return self._source.shape
            
            def get_nnz(self):
                return self._source.nnz
            
            def get_row_data(self, i):
                return self._source.load_row(i)
        
        mat = MyLazyCSR(my_data_source)
        row_sums = mat.sum(axis=1)  # Uses C++ kernel!
    
    Warning:
        - Performance overhead from Python-C++ boundary crossing
        - GIL held during callbacks (no parallelism in callbacks)
        - Best for I/O-bound scenarios, not compute-intensive
    """
    
    __slots__ = (
        '_handle', '_vtable', '_callbacks',
        '_cached_row_idx', '_cached_values', '_cached_indices',
        '_shape_cache', '_nnz_cache', '_dtype',
        '_closed',
    )
    
    def __init__(self, dtype: str = 'float64'):
        """Initialize callback CSR matrix.
        
        Args:
            dtype: Data type ('float32' or 'float64')
        
        Note:
            Subclasses should call super().__init__() after setting up
            any required instance attributes.
        """
        self._dtype = dtype
        self._closed = False
        
        # Cache for most recent row data
        self._cached_row_idx = -1
        self._cached_values: Optional[np.ndarray] = None
        self._cached_indices: Optional[np.ndarray] = None
        
        # Shape/nnz cache
        self._shape_cache: Optional[Tuple[int, int]] = None
        self._nnz_cache: Optional[int] = None
        
        # Create callbacks and handle
        self._callbacks = self._create_callbacks()
        self._vtable = self._create_vtable()
        
        # Create C++ handle
        handle = CallbackHandle()
        err = kernel_callback.create_callback_csr(
            id(self),  # context = Python object ID
            ctypes.byref(self._vtable),
            ctypes.byref(handle)
        )
        if err != 0:
            raise RuntimeError(f"Failed to create callback CSR: error code {err}")
        self._handle = handle
    
    # =========================================================================
    # Abstract Methods - User Must Implement
    # =========================================================================
    
    @abstractmethod
    def get_shape(self) -> Tuple[int, int]:
        """Return matrix dimensions (rows, cols).
        
        Returns:
            Tuple of (rows, cols)
        """
        ...
    
    @abstractmethod
    def get_nnz(self) -> int:
        """Return total number of non-zero elements.
        
        Returns:
            Total nnz count
        """
        ...
    
    @abstractmethod
    def get_row_data(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return data for row i.
        
        Args:
            i: Row index (0-based)
            
        Returns:
            Tuple of (values, indices) as numpy arrays
            - values: float64 or float32 array of non-zero values
            - indices: int64 array of column indices
        
        Note:
            The returned arrays must remain valid until the next call
            to get_row_data() for the same row index.
        """
        ...
    
    # =========================================================================
    # Optional Methods - User May Override
    # =========================================================================
    
    def get_row_length(self, i: int) -> int:
        """Return number of non-zeros in row i.
        
        Default implementation calls get_row_data(). Override for
        faster length queries without loading data.
        
        Args:
            i: Row index
            
        Returns:
            Number of non-zeros in row i
        """
        values, _ = self.get_row_data(i)
        return len(values)
    
    def prefetch_rows(self, start: int, end: int) -> None:
        """Prefetch rows [start, end) for performance.
        
        Default implementation does nothing. Override to implement
        batch prefetching for better I/O patterns.
        
        Args:
            start: Start row index (inclusive)
            end: End row index (exclusive)
        """
        pass
    
    def release_row(self, i: int) -> None:
        """Release resources for row i.
        
        Default implementation does nothing. Override if you need
        to release memory or other resources after a row is processed.
        
        Args:
            i: Row index
        """
        pass
    
    def close(self) -> None:
        """Close and release all resources.
        
        Override to add custom cleanup. Always call super().close().
        """
        if not self._closed and self._handle:
            kernel_callback.destroy_callback_csr(self._handle)
            self._handle = None
            self._closed = True
    
    # =========================================================================
    # SparseBase/CSRBase Implementation
    # =========================================================================
    
    @property
    def shape(self) -> Tuple[int, int]:
        if self._shape_cache is None:
            self._shape_cache = self.get_shape()
        return self._shape_cache
    
    @property
    def dtype(self) -> str:
        return self._dtype
    
    @property
    def nnz(self) -> int:
        if self._nnz_cache is None:
            self._nnz_cache = self.get_nnz()
        return self._nnz_cache
    
    @property
    def is_view(self) -> bool:
        return True  # Callback matrices are conceptually views
    
    @property
    def is_contiguous(self) -> bool:
        return False  # Data is not contiguous
    
    def row_values(self, i: int) -> Array:
        """Get values for row i."""
        self._ensure_row_cached(i)
        return Array.from_numpy(self._cached_values, copy=False)
    
    def row_indices(self, i: int) -> Array:
        """Get column indices for row i."""
        self._ensure_row_cached(i)
        return Array.from_numpy(self._cached_indices, copy=False)
    
    def row_length(self, i: int) -> int:
        """Get number of non-zeros in row i."""
        return self.get_row_length(i)
    
    # =========================================================================
    # Statistics - Use C++ Kernels
    # =========================================================================
    
    def sum(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute sum along axis using C++ kernel."""
        if axis is None:
            # Total sum: sum all row sums
            row_sums = self.sum(axis=1)
            return float(np.sum(row_sums))
        elif axis == 1:
            # Row sums - use C++ kernel
            output = Array.zeros(self.rows, dtype=self._dtype)
            err = kernel_callback.callback_csr_row_sums(
                self._handle, output.get_pointer()
            )
            if err != 0:
                raise RuntimeError(f"row_sums failed: error code {err}")
            return output.to_numpy()
        elif axis == 0:
            # Column sums - need to iterate
            col_sums = np.zeros(self.cols, dtype=np.float64)
            for i in range(self.rows):
                values, indices = self.get_row_data(i)
                for k, j in enumerate(indices):
                    col_sums[j] += values[k]
            return col_sums
        else:
            raise ValueError(f"Invalid axis: {axis}")
    
    def mean(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute mean along axis using C++ kernel."""
        if axis is None:
            return self.sum(axis=None) / self.size
        elif axis == 1:
            output = Array.zeros(self.rows, dtype=self._dtype)
            err = kernel_callback.callback_csr_row_means(
                self._handle, output.get_pointer()
            )
            if err != 0:
                raise RuntimeError(f"row_means failed: error code {err}")
            return output.to_numpy()
        elif axis == 0:
            return self.sum(axis=0) / self.rows
        else:
            raise ValueError(f"Invalid axis: {axis}")
    
    def var(self, axis: Optional[int] = None, ddof: int = 0) -> Union[float, np.ndarray]:
        """Compute variance along axis using C++ kernel."""
        if axis == 1:
            output = Array.zeros(self.rows, dtype=self._dtype)
            err = kernel_callback.callback_csr_row_variances(
                self._handle, output.get_pointer(), ddof
            )
            if err != 0:
                raise RuntimeError(f"row_variances failed: error code {err}")
            return output.to_numpy()
        else:
            # Fallback to base implementation
            return super().var(axis=axis, ddof=ddof)
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_scipy(self) -> 'spmatrix':
        """Convert to scipy CSR matrix."""
        import scipy.sparse as sp
        
        rows, cols = self.shape
        
        # Collect all data
        all_data = []
        all_indices = []
        indptr = [0]
        
        for i in range(rows):
            values, indices = self.get_row_data(i)
            all_data.append(values)
            all_indices.append(indices)
            indptr.append(indptr[-1] + len(values))
        
        data = np.concatenate(all_data) if all_data else np.array([], dtype=np.float64)
        indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int64)
        indptr = np.array(indptr, dtype=np.int64)
        
        return sp.csr_matrix((data, indices, indptr), shape=(rows, cols))
    
    def copy(self) -> 'SparseBase':
        """Create a materialized copy as SclCSR."""
        from ._csr import SclCSR
        return SclCSR.from_scipy(self.to_scipy())
    
    # =========================================================================
    # Context Manager
    # =========================================================================
    
    def __enter__(self) -> 'CallbackCSR':
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
    
    def __del__(self):
        self.close()
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _ensure_row_cached(self, i: int) -> None:
        """Ensure row i is cached."""
        if i != self._cached_row_idx:
            values, indices = self.get_row_data(i)
            # Ensure correct dtypes
            np_dtype = np.float64 if self._dtype == 'float64' else np.float32
            self._cached_values = np.ascontiguousarray(values, dtype=np_dtype)
            self._cached_indices = np.ascontiguousarray(indices, dtype=np.int64)
            self._cached_row_idx = i
    
    def _create_callbacks(self) -> tuple:
        """Create ctypes callback functions."""
        
        @GetRowsFunc
        def get_rows(ctx):
            return self.shape[0]
        
        @GetColsFunc
        def get_cols(ctx):
            return self.shape[1]
        
        @GetNnzFunc
        def get_nnz_cb(ctx):
            return self.nnz
        
        @GetPrimaryValuesFunc
        def get_primary_values(ctx, i, out_data, out_len):
            try:
                self._ensure_row_cached(i)
                out_data[0] = self._cached_values.ctypes.data_as(POINTER(c_real))
                out_len[0] = len(self._cached_values)
                return 0
            except Exception as e:
                print(f"Error in get_primary_values: {e}")
                return 1
        
        @GetPrimaryIndicesFunc
        def get_primary_indices(ctx, i, out_indices, out_len):
            try:
                self._ensure_row_cached(i)
                out_indices[0] = self._cached_indices.ctypes.data_as(POINTER(c_index))
                out_len[0] = len(self._cached_indices)
                return 0
            except Exception as e:
                print(f"Error in get_primary_indices: {e}")
                return 1
        
        @GetPrimaryLengthFunc
        def get_primary_length(ctx, i):
            return self.get_row_length(i)
        
        @PrefetchRangeFunc
        def prefetch_range(ctx, start, end):
            try:
                self.prefetch_rows(start, end)
                return 0
            except:
                return 1
        
        @ReleasePrimaryFunc
        def release_primary(ctx, i):
            try:
                self.release_row(i)
                return 0
            except:
                return 1
        
        return (get_rows, get_cols, get_nnz_cb, 
                get_primary_values, get_primary_indices,
                get_primary_length, prefetch_range, release_primary)
    
    def _create_vtable(self) -> CallbackVTable:
        """Create VTable structure."""
        return CallbackVTable(
            get_rows=self._callbacks[0],
            get_cols=self._callbacks[1],
            get_nnz=self._callbacks[2],
            get_primary_values=self._callbacks[3],
            get_primary_indices=self._callbacks[4],
            get_primary_length=self._callbacks[5],
            prefetch_range=self._callbacks[6],
            release_primary=self._callbacks[7],
        )


class CallbackCSC(CSCBase):
    """
    Callback-based CSC sparse matrix.
    
    Similar to CallbackCSR but column-oriented. Subclass this and
    implement the required methods for custom column access patterns.
    
    Required Methods to Implement:
        get_shape() -> Tuple[int, int]: Return (rows, cols)
        get_nnz() -> int: Return total non-zero count
        get_col_data(j) -> Tuple[ndarray, ndarray]: Return (values, indices) for column j
    
    Example:
    
        class MyLazyCSC(CallbackCSC):
            def __init__(self, data_source):
                self._source = data_source
                super().__init__()
            
            def get_shape(self):
                return self._source.shape
            
            def get_nnz(self):
                return self._source.nnz
            
            def get_col_data(self, j):
                return self._source.load_column(j)
        
        mat = MyLazyCSC(my_data_source)
        col_sums = mat.sum(axis=0)  # Uses C++ kernel!
    """
    
    __slots__ = (
        '_handle', '_vtable', '_callbacks',
        '_cached_col_idx', '_cached_values', '_cached_indices',
        '_shape_cache', '_nnz_cache', '_dtype',
        '_closed',
    )
    
    def __init__(self, dtype: str = 'float64'):
        """Initialize callback CSC matrix."""
        self._dtype = dtype
        self._closed = False
        
        self._cached_col_idx = -1
        self._cached_values: Optional[np.ndarray] = None
        self._cached_indices: Optional[np.ndarray] = None
        
        self._shape_cache: Optional[Tuple[int, int]] = None
        self._nnz_cache: Optional[int] = None
        
        self._callbacks = self._create_callbacks()
        self._vtable = self._create_vtable()
        
        handle = CallbackHandle()
        err = kernel_callback.create_callback_csc(
            id(self),
            ctypes.byref(self._vtable),
            ctypes.byref(handle)
        )
        if err != 0:
            raise RuntimeError(f"Failed to create callback CSC: error code {err}")
        self._handle = handle
    
    # =========================================================================
    # Abstract Methods - User Must Implement
    # =========================================================================
    
    @abstractmethod
    def get_shape(self) -> Tuple[int, int]:
        """Return matrix dimensions (rows, cols)."""
        ...
    
    @abstractmethod
    def get_nnz(self) -> int:
        """Return total number of non-zero elements."""
        ...
    
    @abstractmethod
    def get_col_data(self, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return data for column j.
        
        Returns:
            Tuple of (values, indices) as numpy arrays
        """
        ...
    
    # =========================================================================
    # Optional Methods
    # =========================================================================
    
    def get_col_length(self, j: int) -> int:
        """Return number of non-zeros in column j."""
        values, _ = self.get_col_data(j)
        return len(values)
    
    def prefetch_cols(self, start: int, end: int) -> None:
        """Prefetch columns [start, end) for performance."""
        pass
    
    def release_col(self, j: int) -> None:
        """Release resources for column j."""
        pass
    
    def close(self) -> None:
        """Close and release all resources."""
        if not self._closed and self._handle:
            kernel_callback.destroy_callback_csc(self._handle)
            self._handle = None
            self._closed = True
    
    # =========================================================================
    # CSCBase Implementation
    # =========================================================================
    
    @property
    def shape(self) -> Tuple[int, int]:
        if self._shape_cache is None:
            self._shape_cache = self.get_shape()
        return self._shape_cache
    
    @property
    def dtype(self) -> str:
        return self._dtype
    
    @property
    def nnz(self) -> int:
        if self._nnz_cache is None:
            self._nnz_cache = self.get_nnz()
        return self._nnz_cache
    
    @property
    def is_view(self) -> bool:
        return True
    
    @property
    def is_contiguous(self) -> bool:
        return False
    
    def col_values(self, j: int) -> Array:
        """Get values for column j."""
        self._ensure_col_cached(j)
        return Array.from_numpy(self._cached_values, copy=False)
    
    def col_indices(self, j: int) -> Array:
        """Get row indices for column j."""
        self._ensure_col_cached(j)
        return Array.from_numpy(self._cached_indices, copy=False)
    
    def col_length(self, j: int) -> int:
        """Get number of non-zeros in column j."""
        return self.get_col_length(j)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def sum(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute sum along axis using C++ kernel."""
        if axis is None:
            col_sums = self.sum(axis=0)
            return float(np.sum(col_sums))
        elif axis == 0:
            output = Array.zeros(self.cols, dtype=self._dtype)
            err = kernel_callback.callback_csc_col_sums(
                self._handle, output.get_pointer()
            )
            if err != 0:
                raise RuntimeError(f"col_sums failed: error code {err}")
            return output.to_numpy()
        elif axis == 1:
            row_sums = np.zeros(self.rows, dtype=np.float64)
            for j in range(self.cols):
                values, indices = self.get_col_data(j)
                for k, i in enumerate(indices):
                    row_sums[i] += values[k]
            return row_sums
        else:
            raise ValueError(f"Invalid axis: {axis}")
    
    def mean(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute mean along axis using C++ kernel."""
        if axis is None:
            return self.sum(axis=None) / self.size
        elif axis == 0:
            output = Array.zeros(self.cols, dtype=self._dtype)
            err = kernel_callback.callback_csc_col_means(
                self._handle, output.get_pointer()
            )
            if err != 0:
                raise RuntimeError(f"col_means failed: error code {err}")
            return output.to_numpy()
        elif axis == 1:
            return self.sum(axis=1) / self.cols
        else:
            raise ValueError(f"Invalid axis: {axis}")
    
    def var(self, axis: Optional[int] = None, ddof: int = 0) -> Union[float, np.ndarray]:
        """Compute variance along axis using C++ kernel."""
        if axis == 0:
            output = Array.zeros(self.cols, dtype=self._dtype)
            err = kernel_callback.callback_csc_col_variances(
                self._handle, output.get_pointer(), ddof
            )
            if err != 0:
                raise RuntimeError(f"col_variances failed: error code {err}")
            return output.to_numpy()
        else:
            return super().var(axis=axis, ddof=ddof)
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_scipy(self) -> 'spmatrix':
        """Convert to scipy CSC matrix."""
        import scipy.sparse as sp
        
        rows, cols = self.shape
        
        all_data = []
        all_indices = []
        indptr = [0]
        
        for j in range(cols):
            values, indices = self.get_col_data(j)
            all_data.append(values)
            all_indices.append(indices)
            indptr.append(indptr[-1] + len(values))
        
        data = np.concatenate(all_data) if all_data else np.array([], dtype=np.float64)
        indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int64)
        indptr = np.array(indptr, dtype=np.int64)
        
        return sp.csc_matrix((data, indices, indptr), shape=(rows, cols))
    
    def copy(self) -> 'SparseBase':
        """Create a materialized copy as SclCSC."""
        from ._csc import SclCSC
        return SclCSC.from_scipy(self.to_scipy())
    
    # =========================================================================
    # Context Manager
    # =========================================================================
    
    def __enter__(self) -> 'CallbackCSC':
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
    
    def __del__(self):
        self.close()
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _ensure_col_cached(self, j: int) -> None:
        """Ensure column j is cached."""
        if j != self._cached_col_idx:
            values, indices = self.get_col_data(j)
            np_dtype = np.float64 if self._dtype == 'float64' else np.float32
            self._cached_values = np.ascontiguousarray(values, dtype=np_dtype)
            self._cached_indices = np.ascontiguousarray(indices, dtype=np.int64)
            self._cached_col_idx = j
    
    def _create_callbacks(self) -> tuple:
        """Create ctypes callback functions."""
        
        @GetRowsFunc
        def get_rows(ctx):
            return self.shape[0]
        
        @GetColsFunc
        def get_cols(ctx):
            return self.shape[1]
        
        @GetNnzFunc
        def get_nnz_cb(ctx):
            return self.nnz
        
        @GetPrimaryValuesFunc
        def get_primary_values(ctx, j, out_data, out_len):
            try:
                self._ensure_col_cached(j)
                out_data[0] = self._cached_values.ctypes.data_as(POINTER(c_real))
                out_len[0] = len(self._cached_values)
                return 0
            except Exception as e:
                print(f"Error in get_primary_values: {e}")
                return 1
        
        @GetPrimaryIndicesFunc
        def get_primary_indices(ctx, j, out_indices, out_len):
            try:
                self._ensure_col_cached(j)
                out_indices[0] = self._cached_indices.ctypes.data_as(POINTER(c_index))
                out_len[0] = len(self._cached_indices)
                return 0
            except Exception as e:
                print(f"Error in get_primary_indices: {e}")
                return 1
        
        @GetPrimaryLengthFunc
        def get_primary_length(ctx, j):
            return self.get_col_length(j)
        
        @PrefetchRangeFunc
        def prefetch_range(ctx, start, end):
            try:
                self.prefetch_cols(start, end)
                return 0
            except:
                return 1
        
        @ReleasePrimaryFunc
        def release_primary(ctx, j):
            try:
                self.release_col(j)
                return 0
            except:
                return 1
        
        return (get_rows, get_cols, get_nnz_cb,
                get_primary_values, get_primary_indices,
                get_primary_length, prefetch_range, release_primary)
    
    def _create_vtable(self) -> CallbackVTable:
        """Create VTable structure."""
        return CallbackVTable(
            get_rows=self._callbacks[0],
            get_cols=self._callbacks[1],
            get_nnz=self._callbacks[2],
            get_primary_values=self._callbacks[3],
            get_primary_indices=self._callbacks[4],
            get_primary_length=self._callbacks[5],
            prefetch_range=self._callbacks[6],
            release_primary=self._callbacks[7],
        )

