"""Smart CSR (Compressed Sparse Row) Matrix.

This module provides SclCSR, a smart sparse matrix class that:
- Transparently manages different backends (Custom, Virtual, Mapped)
- Automatically handles ownership and reference chains
- Optimizes slicing operations based on access patterns
- Provides seamless scipy/numpy interoperability

Design Philosophy:
    Users should not need to worry about memory management or
    backend details. The matrix automatically chooses optimal
    strategies for each operation.

Smart Slicing:
    - Row slice (non-contiguous): Convert to Virtual, immediate slice
    - Column slice: Lazy, mark as view with column mask
    - Contiguous row slice: Zero-copy view when possible

Example:
    >>> # Create from various sources
    >>> mat = SclCSR.from_scipy(scipy_mat)      # Borrowed
    >>> mat = SclCSR.from_dense([[1, 0, 2]])    # Owned
    >>> mat = SclCSR.from_h5ad("data.h5ad")     # Mapped
    >>> 
    >>> # Smart slicing
    >>> view = mat[0:100, :]      # Virtual backend
    >>> subset = mat[[0,10,20], :] # Virtual with indices
    >>> 
    >>> # Operations work regardless of backend
    >>> row_sums = mat.sum(axis=1)
"""

from typing import Tuple, Any, Optional, Union, List, Callable, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

from ._array import Array, zeros, empty, from_list
from ._dtypes import normalize_dtype, validate_dtype
from ._backend import (
    Backend, Ownership, StorageInfo,
    CustomStorage, VirtualStorage, MappedStorage, ChunkInfo
)
from ._ownership import RefChain, OwnershipTracker, ensure_alive
from ._base import CSRBase

if TYPE_CHECKING:
    from ._csc import SclCSC

__all__ = ['SclCSR', 'CSR']


class SclCSR(CSRBase):
    """Smart CSR Sparse Matrix with automatic backend management.
    
    A row-oriented sparse matrix that automatically manages:
    - Data storage (owned, borrowed, or mapped)
    - View hierarchies (reference chains)
    - Slicing strategies (lazy vs immediate)
    - Format conversions (scipy, numpy, anndata)
    
    Backends:
        - CUSTOM: Local arrays, full control
        - VIRTUAL: Zero-copy views for vstack/slicing
        - MAPPED: Memory-mapped files for large data
        
    Ownership:
        - OWNED: We control the memory
        - BORROWED: External memory (scipy), must outlive us
        - VIEW: Derived from another matrix, refs maintained
        
    Attributes:
        shape: Matrix dimensions (rows, cols).
        dtype: Data type ('float32', 'float64').
        nnz: Number of non-zero elements.
        backend: Current backend type.
        ownership: Current ownership model.
        
    Example:
        >>> # From dense
        >>> mat = SclCSR.from_dense([[1, 0, 2], [0, 3, 0]])
        >>> print(mat)  # SclCSR(shape=(2, 3), nnz=3, backend=custom)
        
        >>> # From scipy
        >>> import scipy.sparse as sp
        >>> scipy_mat = sp.csr_matrix([[1, 2], [3, 4]])
        >>> mat = SclCSR.from_scipy(scipy_mat)
        >>> print(mat.ownership)  # Ownership.BORROWED
        
        >>> # Smart slicing
        >>> view = mat[::2, :]  # Non-contiguous -> Virtual
        >>> print(view.backend)  # Backend.VIRTUAL
        
        >>> # Materialize when needed
        >>> owned = view.to_owned()  # Copies data
        >>> print(owned.backend)  # Backend.CUSTOM
    """
    
    __slots__ = (
        '_shape', '_dtype', '_nnz',
        '_backend', '_storage',
        '_ownership', '_ref_chain',
        '_col_mask', '_col_mapping',  # For lazy column slicing
    )
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def __init__(
        self,
        data: Optional[Array] = None,
        indices: Optional[Array] = None,
        indptr: Optional[Array] = None,
        shape: Optional[Tuple[int, int]] = None,
        *,
        row_lengths: Optional[Array] = None,
        backend: Backend = Backend.CUSTOM,
        ownership: Ownership = Ownership.OWNED,
        _storage: Optional[Any] = None,
        _ref_chain: Optional[RefChain] = None,
    ):
        """Initialize SclCSR matrix.
        
        Note:
            Prefer using factory methods (from_dense, from_scipy, etc.)
            instead of direct initialization.
            
        Args:
            data: Non-zero values array.
            indices: Column indices array.
            indptr: Row pointer array.
            shape: Matrix dimensions (rows, cols).
            row_lengths: Optional precomputed row lengths.
            backend: Backend type.
            ownership: Ownership model.
            _storage: Internal storage object.
            _ref_chain: Internal reference chain.
        """
        self._backend = backend
        self._ref_chain = _ref_chain or RefChain()
        self._col_mask = None
        self._col_mapping = None
        
        if _storage is not None:
            # Internal construction with pre-built storage
            self._storage = _storage
            
            if isinstance(_storage, CustomStorage):
                self._shape = shape
                self._dtype = _storage.data.dtype
                self._nnz = _storage.nnz
                self._ownership = OwnershipTracker.owned() if ownership == Ownership.OWNED else OwnershipTracker.borrowed(None)
            elif isinstance(_storage, VirtualStorage):
                self._shape = (_storage.total_primary, _storage.secondary_size)
                self._dtype = _storage.dtype
                self._nnz = -1  # Lazy compute
                self._ownership = OwnershipTracker.view(None)
            elif isinstance(_storage, MappedStorage):
                self._shape = _storage.shape
                self._dtype = _storage.dtype
                self._nnz = _storage.nnz
                self._ownership = OwnershipTracker.owned()
        
        elif data is not None:
            # Standard construction from arrays
            self._validate_arrays(data, indices, indptr, shape)
            
            self._shape = tuple(shape)
            self._dtype = data.dtype
            self._nnz = len(data)
            
            if ownership == Ownership.OWNED:
                self._ownership = OwnershipTracker.owned()
            elif ownership == Ownership.BORROWED:
                self._ownership = OwnershipTracker.borrowed(None)
            else:
                self._ownership = OwnershipTracker.view(None)
            
            # Compute row lengths if not provided
            if row_lengths is None:
                row_lengths = self._compute_row_lengths(indptr, shape[0])
            
            self._storage = CustomStorage(
                data=data,
                indices=indices,
                indptr=indptr,
                primary_lengths=row_lengths,
                ownership=ownership
            )
        else:
            # Empty matrix
            self._shape = shape or (0, 0)
            self._dtype = 'float64'
            self._nnz = 0
            self._ownership = OwnershipTracker.owned()
            self._storage = None
    
    def _validate_arrays(
        self,
        data: Array,
        indices: Array,
        indptr: Array,
        shape: Tuple[int, int]
    ) -> None:
        """Validate array types and dimensions."""
        if data.dtype not in ('float32', 'float64'):
            raise TypeError(f"data must be float32/float64, got {data.dtype}")
        if indices.dtype != 'int64':
            raise TypeError(f"indices must be int64, got {indices.dtype}")
        if indptr.dtype != 'int64':
            raise TypeError(f"indptr must be int64, got {indptr.dtype}")
        
        rows, cols = shape
        if rows < 0 or cols < 0:
            raise ValueError(f"Invalid shape: {shape}")
        if len(indptr) != rows + 1:
            raise ValueError(f"indptr size mismatch: expected {rows+1}, got {len(indptr)}")
        if len(indices) != len(data):
            raise ValueError(f"data/indices size mismatch: {len(data)} vs {len(indices)}")
    
    @staticmethod
    def _compute_row_lengths(indptr: Array, rows: int) -> Array:
        """Compute row lengths from indptr."""
        lengths = zeros(rows, dtype='int64')
        for i in range(rows):
            lengths[i] = indptr[i + 1] - indptr[i]
        return lengths
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix dimensions (rows, cols)."""
        return self._shape
    
    @property
    def rows(self) -> int:
        """Number of rows."""
        return self._shape[0]
    
    @property
    def cols(self) -> int:
        """Number of columns."""
        return self._shape[1]
    
    @property
    def dtype(self) -> str:
        """Data type string."""
        return self._dtype
    
    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        if self._nnz < 0 and isinstance(self._storage, VirtualStorage):
            self._nnz = self._storage.nnz
        return self._nnz
    
    @property
    def backend(self) -> Backend:
        """Current backend type."""
        return self._backend
    
    @property
    def ownership(self) -> Ownership:
        """Current ownership model."""
        if self._ownership.is_owned:
            return Ownership.OWNED
        elif self._ownership.is_borrowed:
            return Ownership.BORROWED
        else:
            return Ownership.VIEW
    
    @property
    def is_owned(self) -> bool:
        """Check if data is owned."""
        return self._ownership.is_owned
    
    @property
    def is_view(self) -> bool:
        """Check if matrix is a view."""
        return self._backend == Backend.VIRTUAL or self._ownership.is_view
    
    @property
    def is_contiguous(self) -> bool:
        """Check if data is contiguous in memory."""
        if self._backend == Backend.CUSTOM:
            return True
        elif self._backend == Backend.VIRTUAL:
            return self._storage.is_contiguous
        return False
    
    @property
    def row_lengths(self) -> Optional[Array]:
        """Row lengths array (if available)."""
        if isinstance(self._storage, CustomStorage):
            return self._storage.primary_lengths
        return None
    
    @property
    def data(self) -> Array:
        """Non-zero values array.
        
        Note:
            For Virtual/Mapped backends, this may trigger materialization.
        """
        self._ensure_custom()
        return self._storage.data
    
    @property
    def indices(self) -> Array:
        """Column indices array."""
        self._ensure_custom()
        return self._storage.indices
    
    @property
    def indptr(self) -> Array:
        """Row pointer array."""
        self._ensure_custom()
        return self._storage.indptr
    
    @property
    def format(self) -> str:
        """Sparse format (always 'csr')."""
        return 'csr'
    
    # =========================================================================
    # CSRBase Interface Implementation
    # =========================================================================
    
    def row_values(self, i: int) -> Array:
        """Get non-zero values for row i.
        
        Args:
            i: Row index (0 <= i < rows)
            
        Returns:
            Array of non-zero values in row i
            
        Note:
            For Virtual/Mapped backends, this may trigger partial materialization.
        """
        if self._backend == Backend.CUSTOM:
            start = int(self._storage.indptr[i])
            end = int(self._storage.indptr[i + 1])
            length = end - start
            
            if length == 0:
                return zeros(0, dtype=self._dtype)
            
            # Create view using numpy slice
            data_np = self._storage.data.to_numpy()
            return Array.from_numpy(data_np[start:end], copy=False)
            
        elif self._backend == Backend.VIRTUAL:
            # For virtual storage, get from the source chunk
            chunk_idx, local_idx = self._storage.get_chunk_for_index(i)
            chunk = self._storage.chunks[chunk_idx]
            
            if chunk.is_identity:
                # Direct access to source
                src = chunk.source
                return src.row_values(local_idx)
            else:
                # Indexed access
                src = chunk.source
                actual_idx = int(chunk.local_indices[local_idx])
                return src.row_values(actual_idx)
        else:
            # Mapped backend - materialize first
            self._ensure_custom()
            return self.row_values(i)
    
    def row_indices(self, i: int) -> Array:
        """Get column indices of non-zeros for row i.
        
        Args:
            i: Row index (0 <= i < rows)
            
        Returns:
            Array of column indices for non-zeros in row i
        """
        if self._backend == Backend.CUSTOM:
            start = int(self._storage.indptr[i])
            end = int(self._storage.indptr[i + 1])
            length = end - start
            
            if length == 0:
                return zeros(0, dtype='int64')
            
            indices_np = self._storage.indices.to_numpy()
            return Array.from_numpy(indices_np[start:end], copy=False)
            
        elif self._backend == Backend.VIRTUAL:
            chunk_idx, local_idx = self._storage.get_chunk_for_index(i)
            chunk = self._storage.chunks[chunk_idx]
            
            if chunk.is_identity:
                src = chunk.source
                return src.row_indices(local_idx)
            else:
                src = chunk.source
                actual_idx = int(chunk.local_indices[local_idx])
                return src.row_indices(actual_idx)
        else:
            self._ensure_custom()
            return self.row_indices(i)
    
    def row_length(self, i: int) -> int:
        """Get number of non-zeros in row i.
        
        Args:
            i: Row index (0 <= i < rows)
            
        Returns:
            Number of non-zero elements in row i
        """
        if self._backend == Backend.CUSTOM:
            return int(self._storage.indptr[i + 1] - self._storage.indptr[i])
        elif self._backend == Backend.VIRTUAL:
            chunk_idx, local_idx = self._storage.get_chunk_for_index(i)
            chunk = self._storage.chunks[chunk_idx]
            
            if chunk.is_identity:
                src = chunk.source
                return src.row_length(local_idx)
            else:
                src = chunk.source
                actual_idx = int(chunk.local_indices[local_idx])
                return src.row_length(actual_idx)
        else:
            self._ensure_custom()
            return self.row_length(i)
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def from_dense(
        cls,
        dense: List[List[float]],
        dtype: str = 'float64'
    ) -> 'SclCSR':
        """Create from dense 2D list.
        
        Args:
            dense: 2D list [rows][cols].
            dtype: Data type ('float32' or 'float64').
            
        Returns:
            New SclCSR matrix with OWNED data.
            
        Example:
            >>> mat = SclCSR.from_dense([[1, 0, 2], [0, 3, 0]])
            >>> print(mat.shape)  # (2, 3)
        """
        validate_dtype(dtype)
        
        if len(dense) == 0:
            return cls.empty(0, 0, 0, dtype)
        
        rows = len(dense)
        cols = len(dense[0])
        
        data_list = []
        indices_list = []
        indptr_list = [0]
        
        for row in dense:
            for j, val in enumerate(row):
                if val != 0.0:
                    data_list.append(val)
                    indices_list.append(j)
            indptr_list.append(len(data_list))
        
        data = from_list(data_list, dtype=dtype)
        indices = from_list(indices_list, dtype='int64')
        indptr = from_list(indptr_list, dtype='int64')
        
        return cls(data, indices, indptr, shape=(rows, cols), ownership=Ownership.OWNED)
    
    @classmethod
    def from_scipy(cls, mat: Any, copy: bool = False) -> 'SclCSR':
        """Create from scipy.sparse.csr_matrix.
        
        By default, borrows scipy's arrays (zero-copy).
        Use copy=True to create owned data.
        
        Args:
            mat: scipy CSR matrix.
            copy: If True, copy data instead of borrowing.
            
        Returns:
            SclCSR matrix (BORROWED unless copy=True).
            
        Warning:
            For BORROWED ownership, the scipy matrix must
            outlive this SclCSR instance!
            
        Example:
            >>> import scipy.sparse as sp
            >>> scipy_mat = sp.csr_matrix([[1, 2], [3, 4]])
            >>> mat = SclCSR.from_scipy(scipy_mat)
            >>> print(mat.ownership)  # Ownership.BORROWED
        """
        try:
            import scipy.sparse as sp
            import numpy as np
        except ImportError:
            raise ImportError("scipy required for from_scipy()")
        
        if not sp.isspmatrix_csr(mat):
            raise TypeError("Expected scipy.sparse.csr_matrix")
        
        # Canonicalize
        mat.sort_indices()
        mat.eliminate_zeros()
        
        # Ensure int64 indices
        if mat.indices.dtype != np.int64:
            mat.indices = mat.indices.astype(np.int64)
        if mat.indptr.dtype != np.int64:
            mat.indptr = mat.indptr.astype(np.int64)
        
        dtype = 'float32' if mat.data.dtype == np.float32 else 'float64'
        
        if copy:
            # Create owned copies
            data = from_list(mat.data.tolist(), dtype=dtype)
            indices = from_list(mat.indices.tolist(), dtype='int64')
            indptr = from_list(mat.indptr.tolist(), dtype='int64')
            ownership = Ownership.OWNED
            source_ref = None
        else:
            # Borrow scipy arrays (zero-copy)
            data = Array.from_buffer(mat.data, dtype, len(mat.data))
            indices = Array.from_buffer(mat.indices, 'int64', len(mat.indices))
            indptr = Array.from_buffer(mat.indptr, 'int64', len(mat.indptr))
            ownership = Ownership.BORROWED
            source_ref = mat  # Keep scipy matrix alive
        
        result = cls(data, indices, indptr, shape=mat.shape, ownership=ownership)
        
        if not copy:
            result._storage._source_ref = source_ref
            result._storage.ownership = Ownership.BORROWED
            result._ownership = OwnershipTracker.borrowed(mat)
            result._ref_chain.add(mat)
        
        return result
    
    @classmethod
    def empty(cls, rows: int, cols: int, nnz: int, dtype: str = 'float64') -> 'SclCSR':
        """Create empty matrix with pre-allocated arrays.
        
        Args:
            rows: Number of rows.
            cols: Number of columns.
            nnz: Number of non-zeros to allocate.
            dtype: Data type.
            
        Returns:
            Empty SclCSR with allocated but uninitialized arrays.
        """
        validate_dtype(dtype)
        
        data = empty(nnz, dtype=dtype)
        indices = empty(nnz, dtype='int64')
        indptr = zeros(rows + 1, dtype='int64')
        
        return cls(data, indices, indptr, shape=(rows, cols), ownership=Ownership.OWNED)
    
    @classmethod
    def zeros(cls, rows: int, cols: int, dtype: str = 'float64') -> 'SclCSR':
        """Create zero matrix (no non-zeros).
        
        Args:
            rows: Number of rows.
            cols: Number of columns.
            dtype: Data type.
            
        Returns:
            Zero SclCSR matrix.
        """
        return cls.empty(rows, cols, 0, dtype)
    
    @classmethod
    def from_arrays(
        cls,
        data: Union[List, Array],
        indices: Union[List, Array],
        indptr: Union[List, Array],
        shape: Tuple[int, int],
        dtype: str = 'float64'
    ) -> 'SclCSR':
        """Create from raw arrays.
        
        Args:
            data: Non-zero values.
            indices: Column indices.
            indptr: Row pointers.
            shape: Matrix dimensions.
            dtype: Data type.
            
        Returns:
            SclCSR matrix.
        """
        if isinstance(data, list):
            data = from_list(data, dtype=dtype)
        if isinstance(indices, list):
            indices = from_list(indices, dtype='int64')
        if isinstance(indptr, list):
            indptr = from_list(indptr, dtype='int64')
        
        return cls(data, indices, indptr, shape, ownership=Ownership.OWNED)
    
    # =========================================================================
    # Virtual Backend Operations
    # =========================================================================
    
    @classmethod
    def _create_virtual(
        cls,
        sources: List['SclCSR'],
        indices_list: Optional[List[Optional[Array]]] = None
    ) -> 'SclCSR':
        """Create virtual matrix from sources.
        
        Internal method for creating Virtual backend matrices.
        
        Args:
            sources: Source matrices.
            indices_list: Optional row indices for each source.
            
        Returns:
            Virtual SclCSR.
        """
        if len(sources) == 0:
            return cls.zeros(0, 0)
        
        # Validate columns match
        cols = sources[0].cols
        dtype = sources[0].dtype
        
        for src in sources[1:]:
            if src.cols != cols:
                raise ValueError(f"Column mismatch: {cols} vs {src.cols}")
            if src.dtype != dtype:
                raise TypeError(f"dtype mismatch: {dtype} vs {src.dtype}")
        
        # Build chunks
        chunks = []
        total_rows = 0
        ref_chain = RefChain()
        
        for i, src in enumerate(sources):
            local_indices = indices_list[i] if indices_list else None
            
            if local_indices is not None:
                length = len(local_indices)
            else:
                length = src.rows
            
            chunks.append(ChunkInfo(
                source=src,
                local_indices=local_indices,
                offset=total_rows,
                length=length
            ))
            
            total_rows += length
            ref_chain.add(src)
        
        storage = VirtualStorage(
            chunks=chunks,
            total_primary=total_rows,
            secondary_size=cols,
            dtype=dtype
        ).flatten()
        
        result = cls(
            _storage=storage,
            shape=(total_rows, cols),
            backend=Backend.VIRTUAL,
            _ref_chain=ref_chain
        )
        result._shape = (total_rows, cols)
        result._dtype = dtype
        result._nnz = -1  # Lazy
        
        return result
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_scipy(self) -> Any:
        """Convert to scipy.sparse.csr_matrix.
        
        Returns:
            scipy CSR matrix.
            
        Note:
            For Virtual backend, this materializes the data.
        """
        try:
            import scipy.sparse as sp
            import numpy as np
        except ImportError:
            raise ImportError("scipy required for to_scipy()")
        
        self._ensure_custom()
        
        np_dtype = np.float32 if self.dtype == 'float32' else np.float64
        
        data_np = np.frombuffer(self._storage.data.tobytes(), dtype=np_dtype)
        indices_np = np.frombuffer(self._storage.indices.tobytes(), dtype=np.int64)
        indptr_np = np.frombuffer(self._storage.indptr.tobytes(), dtype=np.int64)
        
        return sp.csr_matrix((data_np, indices_np, indptr_np), shape=self.shape)
    
    def to_owned(self) -> 'SclCSR':
        """Convert to owned Custom backend.
        
        Creates a deep copy of the data, ensuring full ownership.
        
        Returns:
            SclCSR with OWNED, CUSTOM backend.
        """
        if self._backend == Backend.CUSTOM and self.is_owned:
            return self.copy()
        
        # Materialize and copy
        self._ensure_custom()
        return self.copy()
    
    def tocsc(self) -> 'SclCSC':
        """Convert to CSC format.
        
        Returns:
            SclCSC matrix.
        """
        from ._csc import SclCSC
        
        scipy_csr = self.to_scipy()
        scipy_csc = scipy_csr.tocsc()
        return SclCSC.from_scipy(scipy_csc, copy=True)
    
    def tocsr(self) -> 'SclCSR':
        """Return self (already CSR)."""
        return self
    
    def to_csc(self) -> 'SclCSC':
        """Convert to CSC format (alias for tocsc)."""
        return self.tocsc()
    
    def to_csr(self) -> 'SclCSR':
        """Return self (alias for tocsr)."""
        return self
    
    def copy(self) -> 'SclCSR':
        """Create deep copy.
        
        Returns:
            New SclCSR with copied data.
        """
        self._ensure_custom()
        
        return SclCSR(
            data=self._storage.data.copy(),
            indices=self._storage.indices.copy(),
            indptr=self._storage.indptr.copy(),
            shape=self.shape,
            row_lengths=self._storage.primary_lengths.copy() if self._storage.primary_lengths else None,
            ownership=Ownership.OWNED
        )
    
    def _ensure_custom(self) -> None:
        """Ensure backend is Custom (materialize if needed)."""
        if self._backend == Backend.CUSTOM:
            return
        
        if self._backend == Backend.VIRTUAL:
            self._materialize_virtual()
        elif self._backend == Backend.MAPPED:
            self._load_mapped()
    
    def materialize(self) -> 'SclCSR':
        """Materialize the matrix (ensure all data is in memory).
        
        For Virtual backend, copies data from sources.
        For Mapped backend, loads data from disk.
        For Custom backend, no-op.
        
        Returns:
            self (for chaining).
        """
        self._ensure_custom()
        return self
    
    # =========================================================================
    # Statistical Methods
    # =========================================================================
    
    def sum(self, axis: Optional[int] = None) -> Union[float, 'Array']:
        """Compute the sum of matrix elements.
        
        Args:
            axis: Axis along which to sum.
                - None: Sum all elements
                - 0: Sum along columns (returns array of length cols)
                - 1: Sum along rows (returns array of length rows)
        
        Returns:
            Sum value(s).
        """
        self._ensure_custom()
        
        if axis is None:
            total = 0.0
            for k in range(self.nnz):
                total += self._storage.data[k]
            return total
        
        elif axis == 1:  # Row sums
            result = zeros(self.rows, dtype='float64')
            for i in range(self.rows):
                start = self._storage.indptr[i]
                end = self._storage.indptr[i + 1]
                row_sum = 0.0
                for k in range(start, end):
                    row_sum += self._storage.data[k]
                result[i] = row_sum
            return result
        
        else:  # Column sums (axis=0)
            result = zeros(self.cols, dtype='float64')
            for k in range(self.nnz):
                col_idx = self._storage.indices[k]
                result[col_idx] += self._storage.data[k]
            return result
    
    def mean(self, axis: Optional[int] = None) -> Union[float, 'Array']:
        """Compute the mean of matrix elements.
        
        Args:
            axis: Axis along which to compute mean.
                - None: Mean of all elements
                - 0: Mean along columns
                - 1: Mean along rows
        
        Returns:
            Mean value(s).
        """
        self._ensure_custom()
        
        if axis is None:
            n_total = self.rows * self.cols
            return self.sum() / n_total if n_total > 0 else 0.0
        
        elif axis == 1:  # Row means
            sums = self.sum(axis=1)
            result = zeros(self.rows, dtype='float64')
            for i in range(self.rows):
                result[i] = sums[i] / self.cols if self.cols > 0 else 0.0
            return result
        
        else:  # Column means (axis=0)
            sums = self.sum(axis=0)
            result = zeros(self.cols, dtype='float64')
            for j in range(self.cols):
                result[j] = sums[j] / self.rows if self.rows > 0 else 0.0
            return result
    
    def min(self, axis: Optional[int] = None) -> Union[float, 'Array']:
        """Compute the minimum value of matrix elements.
        
        Args:
            axis: Axis along which to find minimum.
        
        Returns:
            Minimum value(s).
        """
        self._ensure_custom()
        
        if axis is None:
            min_val = 0.0  # Consider implicit zeros
            for k in range(self.nnz):
                val = self._storage.data[k]
                if val < min_val:
                    min_val = val
            # If not all elements are non-zero, min includes 0
            if self.nnz < self.rows * self.cols:
                min_val = min(min_val, 0.0)
            return min_val
        
        elif axis == 1:  # Row minimums
            result = zeros(self.rows, dtype='float64')
            for i in range(self.rows):
                start = self._storage.indptr[i]
                end = self._storage.indptr[i + 1]
                nnz_row = end - start
                
                if nnz_row == 0:
                    result[i] = 0.0
                elif nnz_row < self.cols:
                    # Row has implicit zeros
                    row_min = 0.0
                    for k in range(start, end):
                        val = self._storage.data[k]
                        if val < row_min:
                            row_min = val
                    result[i] = row_min
                else:
                    # Row is dense
                    row_min = self._storage.data[start]
                    for k in range(start + 1, end):
                        val = self._storage.data[k]
                        if val < row_min:
                            row_min = val
                    result[i] = row_min
            return result
        
        else:  # Column minimums (axis=0)
            result = zeros(self.cols, dtype='float64')
            # Initialize with inf to find real min
            import math
            for j in range(self.cols):
                result[j] = math.inf
            
            # Track non-zeros per column
            col_nnz = zeros(self.cols, dtype='int64')
            
            for k in range(self.nnz):
                col_idx = self._storage.indices[k]
                val = self._storage.data[k]
                col_nnz[col_idx] += 1
                if val < result[col_idx]:
                    result[col_idx] = val
            
            # Consider implicit zeros
            for j in range(self.cols):
                if col_nnz[j] < self.rows:
                    result[j] = min(result[j], 0.0)
                elif col_nnz[j] == 0:
                    result[j] = 0.0
            
            return result
    
    def max(self, axis: Optional[int] = None) -> Union[float, 'Array']:
        """Compute the maximum value of matrix elements.
        
        Args:
            axis: Axis along which to find maximum.
        
        Returns:
            Maximum value(s).
        """
        self._ensure_custom()
        
        if axis is None:
            max_val = 0.0  # Consider implicit zeros
            for k in range(self.nnz):
                val = self._storage.data[k]
                if val > max_val:
                    max_val = val
            # If not all elements are non-zero, max includes 0
            if self.nnz < self.rows * self.cols:
                max_val = max(max_val, 0.0)
            return max_val
        
        elif axis == 1:  # Row maximums
            result = zeros(self.rows, dtype='float64')
            for i in range(self.rows):
                start = self._storage.indptr[i]
                end = self._storage.indptr[i + 1]
                nnz_row = end - start
                
                if nnz_row == 0:
                    result[i] = 0.0
                elif nnz_row < self.cols:
                    # Row has implicit zeros
                    row_max = 0.0
                    for k in range(start, end):
                        val = self._storage.data[k]
                        if val > row_max:
                            row_max = val
                    result[i] = row_max
                else:
                    # Row is dense
                    row_max = self._storage.data[start]
                    for k in range(start + 1, end):
                        val = self._storage.data[k]
                        if val > row_max:
                            row_max = val
                    result[i] = row_max
            return result
        
        else:  # Column maximums (axis=0)
            result = zeros(self.cols, dtype='float64')
            import math
            for j in range(self.cols):
                result[j] = -math.inf
            
            col_nnz = zeros(self.cols, dtype='int64')
            
            for k in range(self.nnz):
                col_idx = self._storage.indices[k]
                val = self._storage.data[k]
                col_nnz[col_idx] += 1
                if val > result[col_idx]:
                    result[col_idx] = val
            
            # Consider implicit zeros
            for j in range(self.cols):
                if col_nnz[j] < self.rows:
                    result[j] = max(result[j], 0.0)
                elif col_nnz[j] == 0:
                    result[j] = 0.0
            
            return result
    
    def _materialize_virtual(self) -> None:
        """Materialize Virtual backend to Custom."""
        if not isinstance(self._storage, VirtualStorage):
            return
        
        vstorage = self._storage
        
        # Compute total nnz
        total_nnz = vstorage.nnz
        
        # Allocate output
        new_data = empty(total_nnz, dtype=self.dtype)
        new_indices = empty(total_nnz, dtype='int64')
        new_indptr = zeros(vstorage.total_primary + 1, dtype='int64')
        
        # Copy data from chunks
        data_pos = 0
        row_pos = 0
        
        for chunk in vstorage.chunks:
            src = chunk.source
            
            # Ensure source is materialized
            if hasattr(src, '_ensure_custom'):
                src._ensure_custom()
            
            if chunk.is_identity:
                # Copy all rows
                src_nnz = src.nnz
                for k in range(src_nnz):
                    new_data[data_pos + k] = src.data[k]
                    new_indices[data_pos + k] = src.indices[k]
                
                for i in range(src.rows):
                    new_indptr[row_pos + i + 1] = new_indptr[row_pos + i] + (src.indptr[i + 1] - src.indptr[i])
                
                data_pos += src_nnz
                row_pos += src.rows
            else:
                # Copy selected rows
                for local_i, src_i in enumerate(chunk.local_indices.tolist()):
                    start = src.indptr[src_i]
                    end = src.indptr[src_i + 1]
                    length = end - start
                    
                    for k in range(length):
                        new_data[data_pos] = src.data[start + k]
                        new_indices[data_pos] = src.indices[start + k]
                        data_pos += 1
                    
                    new_indptr[row_pos + 1] = new_indptr[row_pos] + length
                    row_pos += 1
        
        # Update storage
        self._storage = CustomStorage(
            data=new_data,
            indices=new_indices,
            indptr=new_indptr,
            primary_lengths=self._compute_row_lengths(new_indptr, vstorage.total_primary),
            ownership=Ownership.OWNED
        )
        self._backend = Backend.CUSTOM
        self._ownership = OwnershipTracker.owned()
        self._nnz = total_nnz
    
    def _load_mapped(self) -> None:
        """Load Mapped backend to Custom."""
        raise NotImplementedError("Mapped backend loading not yet implemented")
    
    # =========================================================================
    # Slicing Operations
    # =========================================================================
    
    def slice_rows(
        self,
        row_indices: Union[List[int], Array],
        strategy: str = 'auto'
    ) -> 'SclCSR':
        """Extract subset of rows.
        
        Args:
            row_indices: Row indices to extract.
            strategy: Slicing strategy:
                - 'auto': Choose based on sparsity (default)
                - 'virtual': Always create Virtual backend
                - 'copy': Always copy data
                
        Returns:
            SclCSR with selected rows.
            
        Example:
            >>> sub = mat.slice_rows([0, 10, 20, 30])
            >>> print(sub.backend)  # Depends on strategy
        """
        if isinstance(row_indices, list):
            row_indices = from_list(row_indices, dtype='int64')
        
        n_select = len(row_indices)
        
        # Auto strategy: use virtual for sparse selection
        if strategy == 'auto':
            if n_select < self.rows * 0.5:  # Less than 50%
                strategy = 'virtual'
            else:
                strategy = 'copy'
        
        if strategy == 'virtual':
            return self._create_virtual([self], [row_indices])
        else:
            return self._copy_rows(row_indices)
    
    def _copy_rows(self, row_indices: Array) -> 'SclCSR':
        """Copy selected rows to new matrix."""
        self._ensure_custom()
        
        new_rows = len(row_indices)
        
        # Count nnz
        total_nnz = 0
        for i in range(new_rows):
            row_idx = row_indices[i]
            total_nnz += self._storage.primary_lengths[row_idx]
        
        # Allocate
        new_data = empty(total_nnz, dtype=self.dtype)
        new_indices = empty(total_nnz, dtype='int64')
        new_indptr = zeros(new_rows + 1, dtype='int64')
        
        # Copy
        pos = 0
        for i in range(new_rows):
            row_idx = row_indices[i]
            start = self._storage.indptr[row_idx]
            end = self._storage.indptr[row_idx + 1]
            
            for k in range(start, end):
                new_data[pos] = self._storage.data[k]
                new_indices[pos] = self._storage.indices[k]
                pos += 1
            
            new_indptr[i + 1] = pos
        
        return SclCSR(new_data, new_indices, new_indptr, shape=(new_rows, self.cols), ownership=Ownership.OWNED)
    
    def slice_cols(
        self,
        col_indices: Union[List[int], Array],
        lazy: bool = True
    ) -> 'SclCSR':
        """Extract subset of columns.
        
        Args:
            col_indices: Column indices to keep.
            lazy: If True, mark as view without copying.
            
        Returns:
            SclCSR with selected columns.
        """
        if isinstance(col_indices, list):
            col_indices = from_list(col_indices, dtype='int64')
        
        if lazy and self._col_mask is None:
            # First lazy slice: store mask
            result = SclCSR(
                _storage=self._storage,
                shape=(self.rows, len(col_indices)),
                backend=self._backend,
                _ref_chain=RefChain()
            )
            result._ref_chain.add(self)
            result._dtype = self.dtype
            result._nnz = -1  # Unknown until materialized
            
            # Build mask
            result._col_mask = zeros(self.cols, dtype='uint8')
            result._col_mapping = zeros(self.cols, dtype='int64')
            for new_idx in range(len(col_indices)):
                old_idx = col_indices[new_idx]
                result._col_mask[old_idx] = 1
                result._col_mapping[old_idx] = new_idx
            
            result._ownership = OwnershipTracker.view(self)
            return result
        else:
            # Copy with column filter
            return self._copy_with_col_filter(col_indices)
    
    def _copy_with_col_filter(self, col_indices: Array) -> 'SclCSR':
        """Copy with column filtering."""
        self._ensure_custom()
        
        new_cols = len(col_indices)
        
        # Build mask
        col_mask = zeros(self.cols, dtype='uint8')
        col_mapping = zeros(self.cols, dtype='int64')
        for new_idx in range(new_cols):
            old_idx = col_indices[new_idx]
            col_mask[old_idx] = 1
            col_mapping[old_idx] = new_idx
        
        # Count nnz
        total_nnz = 0
        for k in range(self.nnz):
            if col_mask[self._storage.indices[k]] == 1:
                total_nnz += 1
        
        # Allocate
        new_data = empty(total_nnz, dtype=self.dtype)
        new_indices = empty(total_nnz, dtype='int64')
        new_indptr = zeros(self.rows + 1, dtype='int64')
        
        # Copy
        pos = 0
        for i in range(self.rows):
            start = self._storage.indptr[i]
            end = self._storage.indptr[i + 1]
            
            for k in range(start, end):
                col_idx = self._storage.indices[k]
                if col_mask[col_idx] == 1:
                    new_data[pos] = self._storage.data[k]
                    new_indices[pos] = col_mapping[col_idx]
                    pos += 1
            
            new_indptr[i + 1] = pos
        
        return SclCSR(new_data, new_indices, new_indptr, shape=(self.rows, new_cols), ownership=Ownership.OWNED)
    
    def get_row(self, i: int) -> Tuple[Array, Array]:
        """Get row as sparse (indices, values)."""
        self._ensure_custom()
        
        if i < 0 or i >= self.rows:
            raise IndexError(f"Row {i} out of bounds [0, {self.rows})")
        
        start = self._storage.indptr[i]
        end = self._storage.indptr[i + 1]
        length = end - start
        
        if length == 0:
            return empty(0, dtype='int64'), empty(0, dtype=self.dtype)
        
        row_indices = empty(length, dtype='int64')
        row_values = empty(length, dtype=self.dtype)
        
        for k in range(length):
            row_indices[k] = self._storage.indices[start + k]
            row_values[k] = self._storage.data[start + k]
        
        return row_indices, row_values
    
    def get_row_dense(self, i: int) -> Array:
        """Get row as dense array."""
        indices, values = self.get_row(i)
        
        result = zeros(self.cols, dtype=self.dtype)
        for k in range(len(indices)):
            result[indices[k]] = values[k]
        
        return result
    
    # =========================================================================
    # Indexing
    # =========================================================================
    
    def __getitem__(self, key) -> Union['SclCSR', Array, float]:
        """Support mat[i], mat[i, j], mat[i:j], mat[[...], :].
        
        Smart slicing automatically chooses optimal strategy.
        """
        if isinstance(key, int):
            return self.get_row_dense(key)
        
        if isinstance(key, slice):
            start, stop, step = key.indices(self.rows)
            if step == 1:
                indices = list(range(start, stop))
            else:
                indices = list(range(start, stop, step))
            return self.slice_rows(indices)
        
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            
            # mat[i, j]
            if isinstance(row_key, int) and isinstance(col_key, int):
                return self._get_element(row_key, col_key)
            
            # mat[i, :]
            if isinstance(row_key, int) and col_key == slice(None):
                return self.get_row_dense(row_key)
            
            # mat[:, j]
            if row_key == slice(None) and isinstance(col_key, int):
                return self._get_col_dense(col_key)
            
            # mat[i:j, :]
            if isinstance(row_key, slice) and col_key == slice(None):
                start, stop, step = row_key.indices(self.rows)
                if step == 1:
                    indices = list(range(start, stop))
                else:
                    indices = list(range(start, stop, step))
                return self.slice_rows(indices)
            
            # mat[[...], :]
            if isinstance(row_key, (list, Array)) and col_key == slice(None):
                return self.slice_rows(row_key)
            
            # mat[:, [...]]
            if row_key == slice(None) and isinstance(col_key, (list, Array)):
                return self.slice_cols(col_key)
            
            # mat[[...], [...]]
            if isinstance(row_key, (list, Array)) and isinstance(col_key, (list, Array)):
                return self.slice_rows(row_key).slice_cols(col_key)
        
        raise TypeError(f"Invalid index: {type(key)}")
    
    def _get_element(self, i: int, j: int) -> float:
        """Get single element."""
        self._ensure_custom()
        
        if i < 0:
            i += self.rows
        if j < 0:
            j += self.cols
        
        start = self._storage.indptr[i]
        end = self._storage.indptr[i + 1]
        
        for k in range(start, end):
            if self._storage.indices[k] == j:
                return self._storage.data[k]
        return 0.0
    
    def _get_col_dense(self, j: int) -> Array:
        """Get column as dense array."""
        self._ensure_custom()
        
        result = zeros(self.rows, dtype=self.dtype)
        for i in range(self.rows):
            start = self._storage.indptr[i]
            end = self._storage.indptr[i + 1]
            for k in range(start, end):
                if self._storage.indices[k] == j:
                    result[i] = self._storage.data[k]
                    break
        return result
    
    # =========================================================================
    # C API Interface
    # =========================================================================
    
    def get_c_pointers(self) -> Tuple:
        """Get C-compatible pointers for kernel calls.
        
        Returns:
            (data_ptr, indices_ptr, indptr_ptr, lengths_ptr, rows, cols, nnz)
        """
        self._ensure_custom()
        
        ptrs = self._storage.get_pointers()
        return ptrs + (self.rows, self.cols, self.nnz)
    
    # =========================================================================
    # Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return (
            f"SclCSR(shape={self.shape}, nnz={self.nnz}, "
            f"dtype={self.dtype}, backend={self.backend.value})"
        )
    
    def info(self) -> str:
        """Get detailed information string."""
        lines = [
            f"SclCSR Matrix:",
            f"  shape: {self.shape}",
            f"  nnz: {self.nnz}",
            f"  dtype: {self.dtype}",
            f"  backend: {self.backend.value}",
            f"  ownership: {self.ownership.value}",
            f"  is_view: {self.is_view}",
            f"  is_contiguous: {self.is_contiguous}",
        ]
        if isinstance(self._storage, CustomStorage):
            lines.append(f"  memory: {self._storage.nbytes / 1024:.2f} KB")
        if self._ref_chain and self._ref_chain.count > 0:
            lines.append(f"  ref_chain: {self._ref_chain.count} refs")
        return '\n'.join(lines)


# Backward compatibility alias
CSR = SclCSR
