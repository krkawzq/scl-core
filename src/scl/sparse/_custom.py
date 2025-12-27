"""
Custom Storage Sparse Matrices

Provides direct access to CustomStorage-backed sparse matrices.
These are lightweight classes that expose the raw CSR/CSC arrays
without the smart backend management of SclCSR/SclCSC.

Use Cases:
    - Performance-critical code that needs direct array access
    - Interoperability with external systems
    - Building blocks for higher-level abstractions

For most use cases, prefer SclCSR/SclCSC which provide automatic
backend management and more convenience methods.

Example:
    >>> from scl.sparse import CustomCSR, Array
    >>> 
    >>> # Create directly from arrays
    >>> data = Array.from_list([1.0, 2.0, 3.0], dtype='float64')
    >>> indices = Array.from_list([0, 2, 1], dtype='int64')
    >>> indptr = Array.from_list([0, 2, 3], dtype='int64')
    >>> mat = CustomCSR(data, indices, indptr, shape=(2, 3))
    >>> 
    >>> # Access row data directly
    >>> vals, idxs = mat.get_row(0)
    >>> print(mat.row_sums())  # Uses kernel
"""

from typing import Tuple, Optional, Union, TYPE_CHECKING
import numpy as np

from ._base import CSRBase, CSCBase
from ._array import Array, zeros

if TYPE_CHECKING:
    from scipy.sparse import spmatrix

__all__ = ['CustomCSR', 'CustomCSC']


class CustomCSR(CSRBase):
    """
    Custom-storage CSR sparse matrix.
    
    A lightweight CSR matrix that directly wraps raw arrays without
    the smart backend management of SclCSR. This is useful when you
    need direct control over memory and don't need automatic backend switching.
    
    Attributes:
        data: Non-zero values array
        indices: Column indices array
        indptr: Row pointer array
        shape: Matrix dimensions (rows, cols)
        dtype: Data type string
    
    Memory Model:
        Data can be OWNED (we allocated it) or BORROWED (external source).
        When borrowed, the original source must outlive this object.
    
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
    
    __slots__ = ('_data', '_indices', '_indptr', '_shape', '_dtype', '_row_lengths')
    
    def __init__(
        self,
        data: Array,
        indices: Array,
        indptr: Array,
        shape: Tuple[int, int],
        row_lengths: Optional[Array] = None,
        dtype: Optional[str] = None
    ):
        """Initialize CustomCSR from arrays.
        
        Args:
            data: Non-zero values array
            indices: Column indices array  
            indptr: Row pointer array (length = rows + 1)
            shape: Matrix dimensions (rows, cols)
            row_lengths: Optional precomputed row lengths
            dtype: Data type (inferred from data if not provided)
        """
        self._data = data
        self._indices = indices
        self._indptr = indptr
        self._shape = tuple(shape)
        self._dtype = dtype or data.dtype
        
        # Compute or store row lengths
        if row_lengths is not None:
            self._row_lengths = row_lengths
        else:
            self._row_lengths = self._compute_row_lengths()
    
    def _compute_row_lengths(self) -> Array:
        """Compute row lengths from indptr."""
        rows = self._shape[0]
        lengths = zeros(rows, dtype='int64')
        for i in range(rows):
            lengths[i] = self._indptr[i + 1] - self._indptr[i]
        return lengths
    
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
        
        # Create view into data array
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
    
    # =========================================================================
    # Conversion
    # =========================================================================
    
    def to_scipy(self) -> 'spmatrix':
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
            row_lengths=self._row_lengths.copy() if self._row_lengths else None,
            dtype=self._dtype
        )
    
    # =========================================================================
    # C Pointer Access
    # =========================================================================
    
    def get_c_pointers(self) -> Tuple:
        """Get C-compatible pointers for kernel calls.
        
        Returns:
            (data_ptr, indices_ptr, indptr_ptr, lengths_ptr, rows, cols, nnz)
        """
        return (
            self._data.get_pointer(),
            self._indices.get_pointer(),
            self._indptr.get_pointer(),
            self._row_lengths.get_pointer() if self._row_lengths else None,
            self.rows,
            self.cols,
            self.nnz
        )
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def from_scipy(cls, mat, copy: bool = False) -> 'CustomCSR':
        """Create from scipy CSR matrix.
        
        Args:
            mat: scipy.sparse.csr_matrix
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
        else:
            data = Array.from_numpy(mat.data, copy=False)
            indices = Array.from_numpy(mat.indices.astype(np.int64), copy=False)
            indptr = Array.from_numpy(mat.indptr.astype(np.int64), copy=False)
        
        return cls(data, indices, indptr, shape=mat.shape, dtype=dtype)
    
    @classmethod
    def from_dense(cls, dense, dtype: str = 'float64') -> 'CustomCSR':
        """Create from dense 2D array/list.
        
        Args:
            dense: 2D array or list of lists
            dtype: Data type
            
        Returns:
            CustomCSR matrix
        """
        import numpy as np
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


class CustomCSC(CSCBase):
    """
    Custom-storage CSC sparse matrix.
    
    Column-oriented equivalent of CustomCSR. See CustomCSR for details.
    """
    
    __slots__ = ('_data', '_indices', '_indptr', '_shape', '_dtype', '_col_lengths')
    
    def __init__(
        self,
        data: Array,
        indices: Array,
        indptr: Array,
        shape: Tuple[int, int],
        col_lengths: Optional[Array] = None,
        dtype: Optional[str] = None
    ):
        """Initialize CustomCSC from arrays."""
        self._data = data
        self._indices = indices
        self._indptr = indptr
        self._shape = tuple(shape)
        self._dtype = dtype or data.dtype
        
        if col_lengths is not None:
            self._col_lengths = col_lengths
        else:
            self._col_lengths = self._compute_col_lengths()
    
    def _compute_col_lengths(self) -> Array:
        """Compute column lengths from indptr."""
        cols = self._shape[1]
        lengths = zeros(cols, dtype='int64')
        for j in range(cols):
            lengths[j] = self._indptr[j + 1] - self._indptr[j]
        return lengths
    
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
    
    # =========================================================================
    # Conversion
    # =========================================================================
    
    def to_scipy(self) -> 'spmatrix':
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
            col_lengths=self._col_lengths.copy() if self._col_lengths else None,
            dtype=self._dtype
        )
    
    def get_c_pointers(self) -> Tuple:
        """Get C-compatible pointers."""
        return (
            self._data.get_pointer(),
            self._indices.get_pointer(),
            self._indptr.get_pointer(),
            self._col_lengths.get_pointer() if self._col_lengths else None,
            self.rows,
            self.cols,
            self.nnz
        )
    
    @classmethod
    def from_scipy(cls, mat, copy: bool = False) -> 'CustomCSC':
        """Create from scipy CSC matrix."""
        import scipy.sparse as sp
        
        if not sp.isspmatrix_csc(mat):
            mat = mat.tocsc()
        
        dtype = 'float32' if mat.dtype == np.float32 else 'float64'
        
        if copy:
            data = Array.from_numpy(mat.data.copy(), copy=False)
            indices = Array.from_numpy(mat.indices.copy().astype(np.int64), copy=False)
            indptr = Array.from_numpy(mat.indptr.copy().astype(np.int64), copy=False)
        else:
            data = Array.from_numpy(mat.data, copy=False)
            indices = Array.from_numpy(mat.indices.astype(np.int64), copy=False)
            indptr = Array.from_numpy(mat.indptr.astype(np.int64), copy=False)
        
        return cls(data, indices, indptr, shape=mat.shape, dtype=dtype)
    
    @classmethod
    def from_dense(cls, dense, dtype: str = 'float64') -> 'CustomCSC':
        """Create from dense 2D array/list."""
        import numpy as np
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

