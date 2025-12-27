"""
Virtual Sparse Matrices (Zero-Copy Views)

VirtualCSR/CSC are zero-copy views into other sparse matrices.
They cannot be externally constructed - only produced by slicing operations.

C++ Equivalent:
    scl::VirtualSparse<T, IsCSR> from scl/core/sparse.hpp

Memory Layout:
    - data_ptrs[primary]: Pointers to each row/col's values
    - indices_ptrs[primary]: Pointers to each row/col's indices  
    - lengths[primary]: Length of each row/col

Use Cases:
    - Zero-copy row slicing: mat[::2, :]
    - Zero-copy vstack: vstack([mat1, mat2])
    - Lazy evaluation without data copy

INTERNAL USE ONLY:
    VirtualCSR/CSC are created internally by slicing operations.
    Do not construct directly - use CustomCSR slicing instead.

Example:
    >>> mat = CustomCSR.from_scipy(scipy_mat)
    >>> view = mat[::2, :]  # Returns VirtualCSR (zero-copy)
    >>> view = mat[[0, 5, 10], :]  # Returns VirtualCSR with index map
"""

from typing import Tuple, Optional, Union, List, TYPE_CHECKING, Any
import numpy as np
from ctypes import c_void_p

from ._base import CSRBase, CSCBase
from ._array import Array, zeros, empty

if TYPE_CHECKING:
    from ._custom import CustomCSR, CustomCSC
    from scipy.sparse import spmatrix

__all__ = ['VirtualCSR', 'VirtualCSC']


class _VirtualSparseInternal:
    """Internal marker to prevent external construction."""
    pass


_INTERNAL_KEY = _VirtualSparseInternal()


class VirtualCSR(CSRBase):
    """
    Virtual CSR matrix - zero-copy view with pointer indirection.
    
    INTERNAL USE ONLY: Cannot be constructed externally.
    Created by slicing operations on CustomCSR or other VirtualCSR.
    
    Memory Layout (mirrors C++ VirtualSparse):
        - _data_ptrs: List of pointers to value arrays
        - _indices_ptrs: List of pointers to index arrays
        - _lengths: Array of row lengths
        - _sources: Strong references to source matrices (prevent GC)
        
    C++ Equivalent:
        scl::VirtualSparse<Real, true> aka scl::VirtualCSR
    """
    
    __slots__ = (
        '_data_ptrs', '_indices_ptrs', '_lengths', '_shape', '_dtype',
        '_nnz', '_sources', '_row_data_cache', '_cached_row_idx'
    )
    
    def __init__(
        self,
        data_ptrs: List[Any],
        indices_ptrs: List[Any],
        lengths: Array,
        shape: Tuple[int, int],
        nnz: int,
        sources: List[Any],
        dtype: str = 'float64',
        _internal_key: Any = None
    ):
        """Initialize VirtualCSR (INTERNAL ONLY).
        
        Args:
            data_ptrs: List of ctypes pointers to value arrays per row
            indices_ptrs: List of ctypes pointers to index arrays per row
            lengths: Array of row lengths
            shape: Matrix dimensions
            nnz: Total non-zeros
            sources: List of source matrices to keep alive
            dtype: Data type
            _internal_key: Must be _INTERNAL_KEY to construct
        """
        if _internal_key is not _INTERNAL_KEY:
            raise TypeError(
                "VirtualCSR cannot be constructed directly. "
                "Use slicing on CustomCSR instead: mat[::2, :]"
            )
        
        self._data_ptrs = data_ptrs
        self._indices_ptrs = indices_ptrs
        self._lengths = lengths
        self._shape = tuple(shape)
        self._nnz = nnz
        self._sources = sources  # Keep sources alive
        self._dtype = dtype
        
        # Row cache for repeated access
        self._row_data_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._cached_row_idx = -1
    
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
        return self._nnz
    
    @property
    def format(self) -> str:
        return 'csr'
    
    # =========================================================================
    # CSRBase Interface
    # =========================================================================
    
    def row_values(self, i: int) -> Array:
        """Get non-zero values for row i."""
        self._ensure_row_cached(i)
        values, _ = self._row_data_cache
        return Array.from_numpy(values, copy=False)
    
    def row_indices(self, i: int) -> Array:
        """Get column indices for row i."""
        self._ensure_row_cached(i)
        _, indices = self._row_data_cache
        return Array.from_numpy(indices, copy=False)
    
    def row_length(self, i: int) -> int:
        """Get number of non-zeros in row i."""
        return int(self._lengths[i])
    
    def _ensure_row_cached(self, i: int) -> None:
        """Load row data into cache if not already cached."""
        if self._cached_row_idx == i and self._row_data_cache is not None:
            return
        
        length = int(self._lengths[i])
        
        if length == 0:
            self._row_data_cache = (
                np.array([], dtype=np.float64 if self._dtype == 'float64' else np.float32),
                np.array([], dtype=np.int64)
            )
        else:
            # Read from pointer
            import ctypes
            
            np_dtype = np.float64 if self._dtype == 'float64' else np.float32
            
            # Create numpy array from pointer
            data_ptr = self._data_ptrs[i]
            indices_ptr = self._indices_ptrs[i]
            
            # Use ctypes to create numpy array from pointer
            values = np.ctypeslib.as_array(
                ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_double if self._dtype == 'float64' else ctypes.c_float)),
                shape=(length,)
            ).copy()  # Copy to ensure ownership
            
            indices = np.ctypeslib.as_array(
                ctypes.cast(indices_ptr, ctypes.POINTER(ctypes.c_int64)),
                shape=(length,)
            ).copy()
            
            self._row_data_cache = (values, indices)
        
        self._cached_row_idx = i
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def sum(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute sum along axis."""
        if axis is None:
            total = 0.0
            for i in range(self.rows):
                self._ensure_row_cached(i)
                values, _ = self._row_data_cache
                total += np.sum(values)
            return total
        elif axis == 1:
            result = np.zeros(self.rows, dtype=np.float64)
            for i in range(self.rows):
                self._ensure_row_cached(i)
                values, _ = self._row_data_cache
                result[i] = np.sum(values)
            return result
        else:  # axis == 0
            result = np.zeros(self.cols, dtype=np.float64)
            for i in range(self.rows):
                self._ensure_row_cached(i)
                values, indices = self._row_data_cache
                for k, j in enumerate(indices):
                    result[j] += values[k]
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
        # Materialize for complex operations
        materialized = self.materialize()
        return materialized.var(axis=axis, ddof=ddof)
    
    # =========================================================================
    # Conversion
    # =========================================================================
    
    def materialize(self) -> 'CustomCSR':
        """Materialize to CustomCSR (deep copy).
        
        Converts virtual view to owned contiguous storage.
        
        Returns:
            CustomCSR with copied data
        """
        from ._custom import CustomCSR
        
        # Collect all data
        all_data = []
        all_indices = []
        indptr = [0]
        
        for i in range(self.rows):
            self._ensure_row_cached(i)
            values, indices = self._row_data_cache
            all_data.extend(values.tolist())
            all_indices.extend(indices.tolist())
            indptr.append(len(all_data))
        
        np_dtype = np.float64 if self._dtype == 'float64' else np.float32
        
        return CustomCSR(
            data=Array.from_numpy(np.array(all_data, dtype=np_dtype), copy=False),
            indices=Array.from_numpy(np.array(all_indices, dtype=np.int64), copy=False),
            indptr=Array.from_numpy(np.array(indptr, dtype=np.int64), copy=False),
            shape=self._shape,
            dtype=self._dtype
        )
    
    def to_scipy(self) -> 'spmatrix':
        """Convert to scipy CSR matrix."""
        return self.materialize().to_scipy()
    
    def copy(self) -> 'CustomCSR':
        """Create materialized copy."""
        return self.materialize()
    
    # =========================================================================
    # Internal Factory (for slicing operations)
    # =========================================================================
    
    @classmethod
    def _from_custom_slice(
        cls,
        source: 'CustomCSR',
        row_indices: np.ndarray
    ) -> 'VirtualCSR':
        """Create VirtualCSR from CustomCSR row slice (INTERNAL).
        
        Args:
            source: Source CustomCSR matrix
            row_indices: Array of row indices to include
            
        Returns:
            VirtualCSR view into source
        """
        import ctypes
        
        n_rows = len(row_indices)
        data_ptrs = []
        indices_ptrs = []
        lengths = zeros(n_rows, dtype='int64')
        total_nnz = 0
        
        # Get source arrays
        src_data = source.data.to_numpy()
        src_indices = source.indices.to_numpy()
        src_indptr = source.indptr.to_numpy()
        
        for new_i, old_i in enumerate(row_indices):
            start = int(src_indptr[old_i])
            end = int(src_indptr[old_i + 1])
            length = end - start
            
            lengths[new_i] = length
            total_nnz += length
            
            if length > 0:
                # Get pointer to row data in source
                data_ptrs.append(src_data[start:end].ctypes.data_as(c_void_p))
                indices_ptrs.append(src_indices[start:end].ctypes.data_as(c_void_p))
            else:
                data_ptrs.append(None)
                indices_ptrs.append(None)
        
        return cls(
            data_ptrs=data_ptrs,
            indices_ptrs=indices_ptrs,
            lengths=lengths,
            shape=(n_rows, source.cols),
            nnz=total_nnz,
            sources=[source],  # Keep source alive
            dtype=source.dtype,
            _internal_key=_INTERNAL_KEY
        )
    
    @classmethod
    def _from_vstack(
        cls,
        matrices: List[Union['CustomCSR', 'VirtualCSR']]
    ) -> 'VirtualCSR':
        """Create VirtualCSR from vertical stack (INTERNAL).
        
        Args:
            matrices: List of CSR matrices to stack
            
        Returns:
            VirtualCSR representing stacked matrices
        """
        if not matrices:
            raise ValueError("Cannot vstack empty list")
        
        cols = matrices[0].cols
        dtype = matrices[0].dtype
        
        for mat in matrices:
            if mat.cols != cols:
                raise ValueError(f"Column mismatch: {mat.cols} vs {cols}")
        
        # Collect all rows
        data_ptrs = []
        indices_ptrs = []
        all_lengths = []
        total_nnz = 0
        total_rows = 0
        sources = list(matrices)  # Keep all sources alive
        
        for mat in matrices:
            if isinstance(mat, VirtualCSR):
                # Flatten virtual
                data_ptrs.extend(mat._data_ptrs)
                indices_ptrs.extend(mat._indices_ptrs)
                all_lengths.extend(mat._lengths.to_numpy().tolist())
                total_nnz += mat.nnz
                sources.extend(mat._sources)
            else:
                # CustomCSR - extract row pointers
                src_data = mat.data.to_numpy()
                src_indices = mat.indices.to_numpy()
                src_indptr = mat.indptr.to_numpy()
                
                for i in range(mat.rows):
                    start = int(src_indptr[i])
                    end = int(src_indptr[i + 1])
                    length = end - start
                    
                    all_lengths.append(length)
                    total_nnz += length
                    
                    if length > 0:
                        data_ptrs.append(src_data[start:end].ctypes.data_as(c_void_p))
                        indices_ptrs.append(src_indices[start:end].ctypes.data_as(c_void_p))
                    else:
                        data_ptrs.append(None)
                        indices_ptrs.append(None)
            
            total_rows += mat.rows
        
        lengths = Array.from_numpy(np.array(all_lengths, dtype=np.int64), copy=False)
        
        return cls(
            data_ptrs=data_ptrs,
            indices_ptrs=indices_ptrs,
            lengths=lengths,
            shape=(total_rows, cols),
            nnz=total_nnz,
            sources=sources,
            dtype=dtype,
            _internal_key=_INTERNAL_KEY
        )


class VirtualCSC(CSCBase):
    """
    Virtual CSC matrix - zero-copy view with pointer indirection.
    
    INTERNAL USE ONLY: Cannot be constructed externally.
    Created by slicing operations on CustomCSC or other VirtualCSC.
    
    C++ Equivalent:
        scl::VirtualSparse<Real, false> aka scl::VirtualCSC
    """
    
    __slots__ = (
        '_data_ptrs', '_indices_ptrs', '_lengths', '_shape', '_dtype',
        '_nnz', '_sources', '_col_data_cache', '_cached_col_idx'
    )
    
    def __init__(
        self,
        data_ptrs: List[Any],
        indices_ptrs: List[Any],
        lengths: Array,
        shape: Tuple[int, int],
        nnz: int,
        sources: List[Any],
        dtype: str = 'float64',
        _internal_key: Any = None
    ):
        """Initialize VirtualCSC (INTERNAL ONLY)."""
        if _internal_key is not _INTERNAL_KEY:
            raise TypeError(
                "VirtualCSC cannot be constructed directly. "
                "Use slicing on CustomCSC instead: mat[:, ::2]"
            )
        
        self._data_ptrs = data_ptrs
        self._indices_ptrs = indices_ptrs
        self._lengths = lengths
        self._shape = tuple(shape)
        self._nnz = nnz
        self._sources = sources
        self._dtype = dtype
        
        self._col_data_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._cached_col_idx = -1
    
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
        return self._nnz
    
    @property
    def format(self) -> str:
        return 'csc'
    
    # =========================================================================
    # CSCBase Interface
    # =========================================================================
    
    def col_values(self, j: int) -> Array:
        """Get non-zero values for column j."""
        self._ensure_col_cached(j)
        values, _ = self._col_data_cache
        return Array.from_numpy(values, copy=False)
    
    def col_indices(self, j: int) -> Array:
        """Get row indices for column j."""
        self._ensure_col_cached(j)
        _, indices = self._col_data_cache
        return Array.from_numpy(indices, copy=False)
    
    def col_length(self, j: int) -> int:
        """Get number of non-zeros in column j."""
        return int(self._lengths[j])
    
    def _ensure_col_cached(self, j: int) -> None:
        """Load column data into cache."""
        if self._cached_col_idx == j and self._col_data_cache is not None:
            return
        
        length = int(self._lengths[j])
        
        if length == 0:
            self._col_data_cache = (
                np.array([], dtype=np.float64 if self._dtype == 'float64' else np.float32),
                np.array([], dtype=np.int64)
            )
        else:
            import ctypes
            
            np_dtype = np.float64 if self._dtype == 'float64' else np.float32
            
            data_ptr = self._data_ptrs[j]
            indices_ptr = self._indices_ptrs[j]
            
            values = np.ctypeslib.as_array(
                ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_double if self._dtype == 'float64' else ctypes.c_float)),
                shape=(length,)
            ).copy()
            
            indices = np.ctypeslib.as_array(
                ctypes.cast(indices_ptr, ctypes.POINTER(ctypes.c_int64)),
                shape=(length,)
            ).copy()
            
            self._col_data_cache = (values, indices)
        
        self._cached_col_idx = j
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def sum(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute sum along axis."""
        if axis is None:
            total = 0.0
            for j in range(self.cols):
                self._ensure_col_cached(j)
                values, _ = self._col_data_cache
                total += np.sum(values)
            return total
        elif axis == 0:
            result = np.zeros(self.cols, dtype=np.float64)
            for j in range(self.cols):
                self._ensure_col_cached(j)
                values, _ = self._col_data_cache
                result[j] = np.sum(values)
            return result
        else:  # axis == 1
            result = np.zeros(self.rows, dtype=np.float64)
            for j in range(self.cols):
                self._ensure_col_cached(j)
                values, indices = self._col_data_cache
                for k, i in enumerate(indices):
                    result[i] += values[k]
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
        materialized = self.materialize()
        return materialized.var(axis=axis, ddof=ddof)
    
    # =========================================================================
    # Conversion
    # =========================================================================
    
    def materialize(self) -> 'CustomCSC':
        """Materialize to CustomCSC (deep copy)."""
        from ._custom import CustomCSC
        
        all_data = []
        all_indices = []
        indptr = [0]
        
        for j in range(self.cols):
            self._ensure_col_cached(j)
            values, indices = self._col_data_cache
            all_data.extend(values.tolist())
            all_indices.extend(indices.tolist())
            indptr.append(len(all_data))
        
        np_dtype = np.float64 if self._dtype == 'float64' else np.float32
        
        return CustomCSC(
            data=Array.from_numpy(np.array(all_data, dtype=np_dtype), copy=False),
            indices=Array.from_numpy(np.array(all_indices, dtype=np.int64), copy=False),
            indptr=Array.from_numpy(np.array(indptr, dtype=np.int64), copy=False),
            shape=self._shape,
            dtype=self._dtype
        )
    
    def to_scipy(self) -> 'spmatrix':
        """Convert to scipy CSC matrix."""
        return self.materialize().to_scipy()
    
    def copy(self) -> 'CustomCSC':
        """Create materialized copy."""
        return self.materialize()
    
    @classmethod
    def _from_custom_slice(
        cls,
        source: 'CustomCSC',
        col_indices: np.ndarray
    ) -> 'VirtualCSC':
        """Create VirtualCSC from CustomCSC column slice (INTERNAL)."""
        import ctypes
        
        n_cols = len(col_indices)
        data_ptrs = []
        indices_ptrs = []
        lengths = zeros(n_cols, dtype='int64')
        total_nnz = 0
        
        src_data = source.data.to_numpy()
        src_indices = source.indices.to_numpy()
        src_indptr = source.indptr.to_numpy()
        
        for new_j, old_j in enumerate(col_indices):
            start = int(src_indptr[old_j])
            end = int(src_indptr[old_j + 1])
            length = end - start
            
            lengths[new_j] = length
            total_nnz += length
            
            if length > 0:
                data_ptrs.append(src_data[start:end].ctypes.data_as(c_void_p))
                indices_ptrs.append(src_indices[start:end].ctypes.data_as(c_void_p))
            else:
                data_ptrs.append(None)
                indices_ptrs.append(None)
        
        return cls(
            data_ptrs=data_ptrs,
            indices_ptrs=indices_ptrs,
            lengths=lengths,
            shape=(source.rows, n_cols),
            nnz=total_nnz,
            sources=[source],
            dtype=source.dtype,
            _internal_key=_INTERNAL_KEY
        )
    
    @classmethod
    def _from_hstack(
        cls,
        matrices: List[Union['CustomCSC', 'VirtualCSC']]
    ) -> 'VirtualCSC':
        """Create VirtualCSC from horizontal stack (INTERNAL)."""
        if not matrices:
            raise ValueError("Cannot hstack empty list")
        
        rows = matrices[0].rows
        dtype = matrices[0].dtype
        
        for mat in matrices:
            if mat.rows != rows:
                raise ValueError(f"Row mismatch: {mat.rows} vs {rows}")
        
        data_ptrs = []
        indices_ptrs = []
        all_lengths = []
        total_nnz = 0
        total_cols = 0
        sources = list(matrices)
        
        for mat in matrices:
            if isinstance(mat, VirtualCSC):
                data_ptrs.extend(mat._data_ptrs)
                indices_ptrs.extend(mat._indices_ptrs)
                all_lengths.extend(mat._lengths.to_numpy().tolist())
                total_nnz += mat.nnz
                sources.extend(mat._sources)
            else:
                src_data = mat.data.to_numpy()
                src_indices = mat.indices.to_numpy()
                src_indptr = mat.indptr.to_numpy()
                
                for j in range(mat.cols):
                    start = int(src_indptr[j])
                    end = int(src_indptr[j + 1])
                    length = end - start
                    
                    all_lengths.append(length)
                    total_nnz += length
                    
                    if length > 0:
                        data_ptrs.append(src_data[start:end].ctypes.data_as(c_void_p))
                        indices_ptrs.append(src_indices[start:end].ctypes.data_as(c_void_p))
                    else:
                        data_ptrs.append(None)
                        indices_ptrs.append(None)
            
            total_cols += mat.cols
        
        lengths = Array.from_numpy(np.array(all_lengths, dtype=np.int64), copy=False)
        
        return cls(
            data_ptrs=data_ptrs,
            indices_ptrs=indices_ptrs,
            lengths=lengths,
            shape=(rows, total_cols),
            nnz=total_nnz,
            sources=sources,
            dtype=dtype,
            _internal_key=_INTERNAL_KEY
        )

