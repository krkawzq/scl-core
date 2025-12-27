"""
Custom Storage Sparse Matrices

Directly exposes CustomSparse data structures - the foundation of sparse
matrix storage with contiguous arrays (data, indices, indptr).

C++ Equivalent:
    scl::CustomSparse<T, IsCSR> from scl/core/sparse.hpp

Memory Layout:
    - data[nnz]: Non-zero values
    - indices[nnz]: Secondary dimension indices (col for CSR, row for CSC)
    - indptr[primary+1]: Cumulative offsets

Use Cases:
    - Direct construction from raw arrays
    - Interoperability with scipy.sparse
    - Building block for VirtualCSR/CSC (slicing)
    - High-performance kernels that need direct array access

For most use cases, prefer SclCSR/SclCSC which provide automatic
backend management and more convenience methods.

Example:
    >>> from scl.sparse import CustomCSR, Array
    >>> 
    >>> # Create directly from arrays
    >>> data = Array.from_numpy(np.array([1.0, 2.0, 3.0]))
    >>> indices = Array.from_numpy(np.array([0, 2, 1], dtype=np.int64))
    >>> indptr = Array.from_numpy(np.array([0, 2, 3], dtype=np.int64))
    >>> mat = CustomCSR(data, indices, indptr, shape=(2, 3))
    >>> 
    >>> # Access row data directly
    >>> vals, idxs = mat.get_row(0)
    >>> print(mat.sum(axis=1))
"""

from typing import Tuple, Optional, Union, TYPE_CHECKING, overload
import numpy as np

from ._base import CSRBase, CSCBase
from ._array import Array, zeros

if TYPE_CHECKING:
    from scipy.sparse import spmatrix, csr_matrix, csc_matrix

__all__ = ['CustomCSR', 'CustomCSC']


class CustomCSR(CSRBase):
    """
    Custom-storage CSR sparse matrix.
    
    A CSR matrix that directly wraps contiguous arrays without smart 
    backend management. Maps to C++ CustomSparse<T, true>.
    
    Attributes:
        data: Non-zero values array (nnz elements)
        indices: Column indices array (nnz elements)
        indptr: Row pointer array (rows + 1 elements)
        shape: Matrix dimensions (rows, cols)
        dtype: Data type string ('float32' or 'float64')
    
    Memory Model:
        Data can be OWNED (we allocated it) or BORROWED (external source).
        When borrowed, the original source must outlive this object.
        
    C++ Equivalent:
        scl::CustomSparse<Real, true> aka scl::CustomCSR
    
    Example:
        >>> # From arrays
        >>> mat = CustomCSR(data, indices, indptr, shape=(100, 50))
        >>> 
        >>> # From scipy (borrowed)
        >>> mat = CustomCSR.from_scipy(scipy_csr)
        >>> 
        >>> # Direct row access
        >>> values, indices = mat.get_row(0)
    """
    
    __slots__ = ('_data', '_indices', '_indptr', '_shape', '_dtype', '_source_ref')
    
    def __init__(
        self,
        data: Array,
        indices: Array,
        indptr: Array,
        shape: Tuple[int, int],
        dtype: Optional[str] = None,
        _source_ref: Optional[object] = None
    ):
        """Initialize CustomCSR from arrays.
        
        Args:
            data: Non-zero values array
            indices: Column indices array  
            indptr: Row pointer array (length = rows + 1)
            shape: Matrix dimensions (rows, cols)
            dtype: Data type (inferred from data if not provided)
            _source_ref: Reference to borrowed source (internal use)
        """
        self._data = data
        self._indices = indices
        self._indptr = indptr
        self._shape = tuple(shape)
        self._dtype = dtype or data.dtype
        self._source_ref = _source_ref  # Keep source alive for borrowed data
        
        # Validation
        if len(indptr) != shape[0] + 1:
            raise ValueError(f"indptr length {len(indptr)} != rows + 1 = {shape[0] + 1}")
    
    # =========================================================================
    # Properties (SparseBase)
    # =========================================================================
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape
    
    @property
    def dtype(self) -> str:
        return self._dtype
    
    @property
    def nnz(self) -> int:
        return len(self._data)
    
    @property
    def format(self) -> str:
        return 'csr'
    
    # =========================================================================
    # Array Access Properties
    # =========================================================================
    
    @property
    def data(self) -> Array:
        """Non-zero values array."""
        return self._data
    
    @property
    def indices(self) -> Array:
        """Column indices array."""
        return self._indices
    
    @property
    def indptr(self) -> Array:
        """Row pointer array."""
        return self._indptr
    
    # =========================================================================
    # CSRBase Interface
    # =========================================================================
    
    def row_values(self, i: int) -> Array:
        """Get non-zero values for row i."""
        start = int(self._indptr[i])
        end = int(self._indptr[i + 1])
        length = end - start
        
        if length == 0:
            return zeros(0, dtype=self._dtype)
        
        # Return view into data array
        return Array.from_numpy(
            self._data.to_numpy()[start:end],
            copy=False
        )
    
    def row_indices(self, i: int) -> Array:
        """Get column indices for row i."""
        start = int(self._indptr[i])
        end = int(self._indptr[i + 1])
        length = end - start
        
        if length == 0:
            return zeros(0, dtype='int64')
        
        return Array.from_numpy(
            self._indices.to_numpy()[start:end],
            copy=False
        )
    
    def row_length(self, i: int) -> int:
        """Get number of non-zeros in row i."""
        return int(self._indptr[i + 1] - self._indptr[i])
    
    # =========================================================================
    # Statistics (Override for efficiency)
    # =========================================================================
    
    def sum(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute sum along axis."""
        if axis is None:
            return float(np.sum(self._data.to_numpy()))
        elif axis == 1:
            # Row sums
            result = np.zeros(self.rows, dtype=np.float64)
            data_np = self._data.to_numpy()
            for i in range(self.rows):
                start = int(self._indptr[i])
                end = int(self._indptr[i + 1])
                result[i] = np.sum(data_np[start:end])
            return result
        else:  # axis == 0
            # Column sums
            result = np.zeros(self.cols, dtype=np.float64)
            data_np = self._data.to_numpy()
            indices_np = self._indices.to_numpy()
            for k in range(self.nnz):
                result[indices_np[k]] += data_np[k]
            return result
    
    def mean(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute mean along axis."""
        if axis is None:
            return self.sum() / self.size
        elif axis == 1:
            return self.sum(axis=1) / self.cols
        else:
            return self.sum(axis=0) / self.rows
    
    def var(self, axis: Optional[int] = None, ddof: int = 1) -> Union[float, np.ndarray]:
        """Compute variance along axis."""
        if axis is None:
            data = self._data.to_numpy()
            return float(np.var(data, ddof=ddof))
        elif axis == 1:
            # Row variances
            result = np.zeros(self.rows, dtype=np.float64)
            data_np = self._data.to_numpy()
            indptr_np = self._indptr.to_numpy()
            
            for i in range(self.rows):
                start = int(indptr_np[i])
                end = int(indptr_np[i + 1])
                row_data = data_np[start:end]
                
                # Include zeros in variance calculation
                n_zeros = self.cols - len(row_data)
                if len(row_data) + n_zeros > ddof:
                    mean_val = np.sum(row_data) / self.cols
                    sq_diff = np.sum((row_data - mean_val) ** 2) + n_zeros * mean_val ** 2
                    result[i] = sq_diff / (self.cols - ddof)
            return result
        else:
            # Column variances (similar logic)
            result = np.zeros(self.cols, dtype=np.float64)
            data_np = self._data.to_numpy()
            indices_np = self._indices.to_numpy()
            
            col_sum = np.zeros(self.cols)
            col_sq_sum = np.zeros(self.cols)
            col_count = np.zeros(self.cols, dtype=np.int64)
            
            for k in range(self.nnz):
                j = indices_np[k]
                v = data_np[k]
                col_sum[j] += v
                col_sq_sum[j] += v * v
                col_count[j] += 1
            
            for j in range(self.cols):
                n = self.rows
                if n > ddof:
                    mean_val = col_sum[j] / n
                    # Include zeros
                    n_zeros = n - col_count[j]
                    sq_diff = col_sq_sum[j] - 2 * mean_val * col_sum[j] + n * mean_val ** 2
                    result[j] = sq_diff / (n - ddof)
            return result
    
    # =========================================================================
    # Conversion
    # =========================================================================
    
    def to_scipy(self) -> 'csr_matrix':
        """Convert to scipy CSR matrix."""
        import scipy.sparse as sp
        
        np_dtype = np.float32 if self._dtype == 'float32' else np.float64
        return sp.csr_matrix(
            (self._data.to_numpy().astype(np_dtype),
             self._indices.to_numpy().astype(np.int64),
             self._indptr.to_numpy().astype(np.int64)),
            shape=self._shape
        )
    
    def copy(self) -> 'CustomCSR':
        """Create deep copy."""
        return CustomCSR(
            data=self._data.copy(),
            indices=self._indices.copy(),
            indptr=self._indptr.copy(),
            shape=self._shape,
            dtype=self._dtype
        )
    
    # =========================================================================
    # C Pointer Access (for kernel calls)
    # =========================================================================
    
    def get_c_pointers(self) -> Tuple:
        """Get C-compatible pointers for kernel calls.
        
        Returns:
            (data_ptr, indices_ptr, indptr_ptr, rows, cols, nnz)
        """
        return (
            self._data.get_pointer(),
            self._indices.get_pointer(),
            self._indptr.get_pointer(),
            self.rows,
            self.cols,
            self.nnz
        )
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def from_scipy(cls, mat: 'spmatrix', copy: bool = False) -> 'CustomCSR':
        """Create from scipy CSR matrix.
        
        Args:
            mat: scipy.sparse.csr_matrix (or convertible)
            copy: If True, copy data; otherwise borrow
            
        Returns:
            CustomCSR matrix
        """
        import scipy.sparse as sp
        
        if not sp.isspmatrix_csr(mat):
            mat = mat.tocsr()
        
        dtype = 'float32' if mat.dtype == np.float32 else 'float64'
        
        if copy:
            data = Array.from_numpy(mat.data.copy(), copy=False)
            indices = Array.from_numpy(mat.indices.copy().astype(np.int64), copy=False)
            indptr = Array.from_numpy(mat.indptr.copy().astype(np.int64), copy=False)
            source_ref = None
        else:
            # Borrow - keep reference to original
            data = Array.from_numpy(mat.data, copy=False)
            indices = Array.from_numpy(mat.indices.astype(np.int64), copy=False)
            indptr = Array.from_numpy(mat.indptr.astype(np.int64), copy=False)
            source_ref = mat
        
        return cls(data, indices, indptr, shape=mat.shape, dtype=dtype, 
                   _source_ref=source_ref)
    
    @classmethod
    def from_dense(cls, dense, dtype: str = 'float64') -> 'CustomCSR':
        """Create from dense 2D array/list.
        
        Args:
            dense: 2D array or list of lists
            dtype: Data type
            
        Returns:
            CustomCSR matrix
        """
        dense = np.asarray(dense, dtype=np.float64 if dtype == 'float64' else np.float32)
        
        rows, cols = dense.shape
        data_list = []
        indices_list = []
        indptr_list = [0]
        
        for i in range(rows):
            for j in range(cols):
                if dense[i, j] != 0:
                    data_list.append(dense[i, j])
                    indices_list.append(j)
            indptr_list.append(len(data_list))
        
        data = Array.from_numpy(np.array(data_list, dtype=dense.dtype), copy=False)
        indices = Array.from_numpy(np.array(indices_list, dtype=np.int64), copy=False)
        indptr = Array.from_numpy(np.array(indptr_list, dtype=np.int64), copy=False)
        
        return cls(data, indices, indptr, shape=(rows, cols), dtype=dtype)
    
    @classmethod  
    def from_arrays(
        cls,
        data: np.ndarray,
        indices: np.ndarray, 
        indptr: np.ndarray,
        shape: Tuple[int, int],
        copy: bool = True
    ) -> 'CustomCSR':
        """Create from numpy arrays.
        
        Args:
            data: Non-zero values (nnz,)
            indices: Column indices (nnz,)
            indptr: Row pointers (rows + 1,)
            shape: Matrix shape (rows, cols)
            copy: Whether to copy arrays
            
        Returns:
            CustomCSR matrix
        """
        dtype = 'float32' if data.dtype == np.float32 else 'float64'
        
        return cls(
            data=Array.from_numpy(data, copy=copy),
            indices=Array.from_numpy(indices.astype(np.int64), copy=copy),
            indptr=Array.from_numpy(indptr.astype(np.int64), copy=copy),
            shape=shape,
            dtype=dtype
        )


class CustomCSC(CSCBase):
    """
    Custom-storage CSC sparse matrix.
    
    Column-oriented equivalent of CustomCSR. Maps to C++ CustomSparse<T, false>.
    
    C++ Equivalent:
        scl::CustomSparse<Real, false> aka scl::CustomCSC
    """
    
    __slots__ = ('_data', '_indices', '_indptr', '_shape', '_dtype', '_source_ref')
    
    def __init__(
        self,
        data: Array,
        indices: Array,
        indptr: Array,
        shape: Tuple[int, int],
        dtype: Optional[str] = None,
        _source_ref: Optional[object] = None
    ):
        """Initialize CustomCSC from arrays."""
        self._data = data
        self._indices = indices
        self._indptr = indptr
        self._shape = tuple(shape)
        self._dtype = dtype or data.dtype
        self._source_ref = _source_ref
        
        if len(indptr) != shape[1] + 1:
            raise ValueError(f"indptr length {len(indptr)} != cols + 1 = {shape[1] + 1}")
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape
    
    @property
    def dtype(self) -> str:
        return self._dtype
    
    @property
    def nnz(self) -> int:
        return len(self._data)
    
    @property
    def format(self) -> str:
        return 'csc'
    
    @property
    def data(self) -> Array:
        return self._data
    
    @property
    def indices(self) -> Array:
        return self._indices
    
    @property
    def indptr(self) -> Array:
        return self._indptr
    
    # =========================================================================
    # CSCBase Interface
    # =========================================================================
    
    def col_values(self, j: int) -> Array:
        """Get non-zero values for column j."""
        start = int(self._indptr[j])
        end = int(self._indptr[j + 1])
        
        if end == start:
            return zeros(0, dtype=self._dtype)
        
        return Array.from_numpy(
            self._data.to_numpy()[start:end],
            copy=False
        )
    
    def col_indices(self, j: int) -> Array:
        """Get row indices for column j."""
        start = int(self._indptr[j])
        end = int(self._indptr[j + 1])
        
        if end == start:
            return zeros(0, dtype='int64')
        
        return Array.from_numpy(
            self._indices.to_numpy()[start:end],
            copy=False
        )
    
    def col_length(self, j: int) -> int:
        """Get number of non-zeros in column j."""
        return int(self._indptr[j + 1] - self._indptr[j])
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def sum(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute sum along axis."""
        if axis is None:
            return float(np.sum(self._data.to_numpy()))
        elif axis == 0:
            # Column sums
            result = np.zeros(self.cols, dtype=np.float64)
            data_np = self._data.to_numpy()
            for j in range(self.cols):
                start = int(self._indptr[j])
                end = int(self._indptr[j + 1])
                result[j] = np.sum(data_np[start:end])
            return result
        else:  # axis == 1
            # Row sums
            result = np.zeros(self.rows, dtype=np.float64)
            data_np = self._data.to_numpy()
            indices_np = self._indices.to_numpy()
            for k in range(self.nnz):
                result[indices_np[k]] += data_np[k]
            return result
    
    def mean(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute mean along axis."""
        if axis is None:
            return self.sum() / self.size
        elif axis == 0:
            return self.sum(axis=0) / self.rows
        else:
            return self.sum(axis=1) / self.cols
    
    def var(self, axis: Optional[int] = None, ddof: int = 1) -> Union[float, np.ndarray]:
        """Compute variance along axis."""
        # Similar to CustomCSR but swapped axes
        if axis is None:
            data = self._data.to_numpy()
            return float(np.var(data, ddof=ddof))
        elif axis == 0:
            # Column variances
            result = np.zeros(self.cols, dtype=np.float64)
            data_np = self._data.to_numpy()
            indptr_np = self._indptr.to_numpy()
            
            for j in range(self.cols):
                start = int(indptr_np[j])
                end = int(indptr_np[j + 1])
                col_data = data_np[start:end]
                
                n_zeros = self.rows - len(col_data)
                if len(col_data) + n_zeros > ddof:
                    mean_val = np.sum(col_data) / self.rows
                    sq_diff = np.sum((col_data - mean_val) ** 2) + n_zeros * mean_val ** 2
                    result[j] = sq_diff / (self.rows - ddof)
            return result
        else:  # axis == 1
            # Row variances
            result = np.zeros(self.rows, dtype=np.float64)
            data_np = self._data.to_numpy()
            indices_np = self._indices.to_numpy()
            
            row_sum = np.zeros(self.rows)
            row_sq_sum = np.zeros(self.rows)
            row_count = np.zeros(self.rows, dtype=np.int64)
            
            for k in range(self.nnz):
                i = indices_np[k]
                v = data_np[k]
                row_sum[i] += v
                row_sq_sum[i] += v * v
                row_count[i] += 1
            
            for i in range(self.rows):
                n = self.cols
                if n > ddof:
                    mean_val = row_sum[i] / n
                    n_zeros = n - row_count[i]
                    sq_diff = row_sq_sum[i] - 2 * mean_val * row_sum[i] + n * mean_val ** 2
                    result[i] = sq_diff / (n - ddof)
            return result
    
    # =========================================================================
    # Conversion
    # =========================================================================
    
    def to_scipy(self) -> 'csc_matrix':
        """Convert to scipy CSC matrix."""
        import scipy.sparse as sp
        
        np_dtype = np.float32 if self._dtype == 'float32' else np.float64
        return sp.csc_matrix(
            (self._data.to_numpy().astype(np_dtype),
             self._indices.to_numpy().astype(np.int64),
             self._indptr.to_numpy().astype(np.int64)),
            shape=self._shape
        )
    
    def copy(self) -> 'CustomCSC':
        """Create deep copy."""
        return CustomCSC(
            data=self._data.copy(),
            indices=self._indices.copy(),
            indptr=self._indptr.copy(),
            shape=self._shape,
            dtype=self._dtype
        )
    
    def get_c_pointers(self) -> Tuple:
        """Get C-compatible pointers."""
        return (
            self._data.get_pointer(),
            self._indices.get_pointer(),
            self._indptr.get_pointer(),
            self.rows,
            self.cols,
            self.nnz
        )
    
    @classmethod
    def from_scipy(cls, mat: 'spmatrix', copy: bool = False) -> 'CustomCSC':
        """Create from scipy CSC matrix."""
        import scipy.sparse as sp
        
        if not sp.isspmatrix_csc(mat):
            mat = mat.tocsc()
        
        dtype = 'float32' if mat.dtype == np.float32 else 'float64'
        
        if copy:
            data = Array.from_numpy(mat.data.copy(), copy=False)
            indices = Array.from_numpy(mat.indices.copy().astype(np.int64), copy=False)
            indptr = Array.from_numpy(mat.indptr.copy().astype(np.int64), copy=False)
            source_ref = None
        else:
            data = Array.from_numpy(mat.data, copy=False)
            indices = Array.from_numpy(mat.indices.astype(np.int64), copy=False)
            indptr = Array.from_numpy(mat.indptr.astype(np.int64), copy=False)
            source_ref = mat
        
        return cls(data, indices, indptr, shape=mat.shape, dtype=dtype,
                   _source_ref=source_ref)
    
    @classmethod
    def from_dense(cls, dense, dtype: str = 'float64') -> 'CustomCSC':
        """Create from dense 2D array/list."""
        dense = np.asarray(dense, dtype=np.float64 if dtype == 'float64' else np.float32)
        
        rows, cols = dense.shape
        data_list = []
        indices_list = []
        indptr_list = [0]
        
        for j in range(cols):
            for i in range(rows):
                if dense[i, j] != 0:
                    data_list.append(dense[i, j])
                    indices_list.append(i)
            indptr_list.append(len(data_list))
        
        data = Array.from_numpy(np.array(data_list, dtype=dense.dtype), copy=False)
        indices = Array.from_numpy(np.array(indices_list, dtype=np.int64), copy=False)
        indptr = Array.from_numpy(np.array(indptr_list, dtype=np.int64), copy=False)
        
        return cls(data, indices, indptr, shape=(rows, cols), dtype=dtype)
    
    @classmethod
    def from_arrays(
        cls,
        data: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: Tuple[int, int],
        copy: bool = True
    ) -> 'CustomCSC':
        """Create from numpy arrays."""
        dtype = 'float32' if data.dtype == np.float32 else 'float64'
        
        return cls(
            data=Array.from_numpy(data, copy=copy),
            indices=Array.from_numpy(indices.astype(np.int64), copy=copy),
            indptr=Array.from_numpy(indptr.astype(np.int64), copy=copy),
            shape=shape,
            dtype=dtype
        )
