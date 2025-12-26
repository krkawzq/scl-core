"""Smart CSC (Compressed Sparse Column) Matrix.

This module provides SclCSC, a smart column-oriented sparse matrix
with automatic backend management, similar to SclCSR but optimized
for column-wise operations.

Design Philosophy:
    Same as SclCSR - transparent backend management, automatic
    ownership tracking, and smart slicing strategies.

Smart Slicing:
    - Column slice (non-contiguous): Convert to Virtual, immediate slice
    - Row slice: Lazy, mark as view with row mask

Example:
    >>> mat = SclCSC.from_scipy(scipy_mat)
    >>> view = mat[:, ::2]  # Non-contiguous columns -> Virtual
    >>> subset = mat[:, [0, 10, 20]]  # Virtual with indices
"""

from typing import Tuple, Any, Optional, Union, List, TYPE_CHECKING

from ._array import Array, zeros, empty, from_list
from ._dtypes import normalize_dtype, validate_dtype
from ._backend import (
    Backend, Ownership, StorageInfo,
    CustomStorage, VirtualStorage, MappedStorage, ChunkInfo
)
from ._ownership import RefChain, OwnershipTracker, ensure_alive

if TYPE_CHECKING:
    from ._csr import SclCSR

__all__ = ['SclCSC', 'CSC']


class SclCSC:
    """Smart CSC Sparse Matrix with automatic backend management.
    
    A column-oriented sparse matrix that automatically manages:
    - Data storage (owned, borrowed, or mapped)
    - View hierarchies (reference chains)
    - Slicing strategies (lazy vs immediate)
    - Format conversions (scipy, numpy, anndata)
    
    Optimal for:
        - Column slicing and iteration
        - Transposed matrix-vector products
        - Feature-wise statistics (genes in scRNA-seq)
        
    Attributes:
        shape: Matrix dimensions (rows, cols).
        dtype: Data type ('float32', 'float64').
        nnz: Number of non-zero elements.
        backend: Current backend type.
        ownership: Current ownership model.
        
    Example:
        >>> mat = SclCSC.from_scipy(scipy_csc_mat)
        >>> col_sums = mat.sum(axis=0)
        >>> subset = mat[:, 0:1000]  # First 1000 columns
    """
    
    __slots__ = (
        '_shape', '_dtype', '_nnz',
        '_backend', '_storage',
        '_ownership', '_ref_chain',
        '_row_mask', '_row_mapping',  # For lazy row slicing
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
        col_lengths: Optional[Array] = None,
        backend: Backend = Backend.CUSTOM,
        ownership: Ownership = Ownership.OWNED,
        _storage: Optional[Any] = None,
        _ref_chain: Optional[RefChain] = None,
    ):
        """Initialize SclCSC matrix.
        
        Note:
            Prefer using factory methods instead of direct initialization.
        """
        self._backend = backend
        self._ref_chain = _ref_chain or RefChain()
        self._row_mask = None
        self._row_mapping = None
        
        if _storage is not None:
            self._storage = _storage
            
            if isinstance(_storage, CustomStorage):
                self._shape = shape
                self._dtype = _storage.data.dtype
                self._nnz = _storage.nnz
                self._ownership = OwnershipTracker.owned() if ownership == Ownership.OWNED else OwnershipTracker.borrowed(None)
            elif isinstance(_storage, VirtualStorage):
                self._shape = (_storage.secondary_size, _storage.total_primary)
                self._dtype = _storage.dtype
                self._nnz = -1
                self._ownership = OwnershipTracker.view(None)
            elif isinstance(_storage, MappedStorage):
                self._shape = _storage.shape
                self._dtype = _storage.dtype
                self._nnz = _storage.nnz
                self._ownership = OwnershipTracker.owned()
        
        elif data is not None:
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
            
            if col_lengths is None:
                col_lengths = self._compute_col_lengths(indptr, shape[1])
            
            self._storage = CustomStorage(
                data=data,
                indices=indices,
                indptr=indptr,
                primary_lengths=col_lengths,
                ownership=ownership
            )
        else:
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
        if len(indptr) != cols + 1:
            raise ValueError(f"indptr size mismatch: expected {cols+1}, got {len(indptr)}")
    
    @staticmethod
    def _compute_col_lengths(indptr: Array, cols: int) -> Array:
        """Compute column lengths from indptr."""
        lengths = zeros(cols, dtype='int64')
        for j in range(cols):
            lengths[j] = indptr[j + 1] - indptr[j]
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
    def col_lengths(self) -> Optional[Array]:
        """Column lengths array (if available)."""
        if isinstance(self._storage, CustomStorage):
            return self._storage.primary_lengths
        return None
    
    @property
    def data(self) -> Array:
        """Non-zero values array."""
        self._ensure_custom()
        return self._storage.data
    
    @property
    def indices(self) -> Array:
        """Row indices array."""
        self._ensure_custom()
        return self._storage.indices
    
    @property
    def indptr(self) -> Array:
        """Column pointer array."""
        self._ensure_custom()
        return self._storage.indptr
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def from_dense(
        cls,
        dense: List[List[float]],
        dtype: str = 'float64'
    ) -> 'SclCSC':
        """Create from dense 2D list.
        
        Args:
            dense: 2D list [rows][cols].
            dtype: Data type.
            
        Returns:
            SclCSC with OWNED data.
        """
        validate_dtype(dtype)
        
        if len(dense) == 0:
            return cls.empty(0, 0, 0, dtype)
        
        rows = len(dense)
        cols = len(dense[0])
        
        data_list = []
        indices_list = []
        indptr_list = [0]
        
        for j in range(cols):
            for i in range(rows):
                val = dense[i][j]
                if val != 0.0:
                    data_list.append(val)
                    indices_list.append(i)
            indptr_list.append(len(data_list))
        
        data = from_list(data_list, dtype=dtype)
        indices = from_list(indices_list, dtype='int64')
        indptr = from_list(indptr_list, dtype='int64')
        
        return cls(data, indices, indptr, shape=(rows, cols), ownership=Ownership.OWNED)
    
    @classmethod
    def from_scipy(cls, mat: Any, copy: bool = False) -> 'SclCSC':
        """Create from scipy.sparse.csc_matrix.
        
        Args:
            mat: scipy CSC matrix.
            copy: If True, copy data instead of borrowing.
            
        Returns:
            SclCSC matrix.
        """
        try:
            import scipy.sparse as sp
            import numpy as np
        except ImportError:
            raise ImportError("scipy required for from_scipy()")
        
        if not sp.isspmatrix_csc(mat):
            raise TypeError("Expected scipy.sparse.csc_matrix")
        
        mat.sort_indices()
        mat.eliminate_zeros()
        
        if mat.indices.dtype != np.int64:
            mat.indices = mat.indices.astype(np.int64)
        if mat.indptr.dtype != np.int64:
            mat.indptr = mat.indptr.astype(np.int64)
        
        dtype = 'float32' if mat.data.dtype == np.float32 else 'float64'
        
        if copy:
            data = from_list(mat.data.tolist(), dtype=dtype)
            indices = from_list(mat.indices.tolist(), dtype='int64')
            indptr = from_list(mat.indptr.tolist(), dtype='int64')
            ownership = Ownership.OWNED
            source_ref = None
        else:
            data = Array.from_buffer(mat.data, dtype, len(mat.data))
            indices = Array.from_buffer(mat.indices, 'int64', len(mat.indices))
            indptr = Array.from_buffer(mat.indptr, 'int64', len(mat.indptr))
            ownership = Ownership.BORROWED
            source_ref = mat
        
        result = cls(data, indices, indptr, shape=mat.shape, ownership=ownership)
        
        if not copy:
            result._storage._source_ref = source_ref
            result._storage.ownership = Ownership.BORROWED
            result._ownership = OwnershipTracker.borrowed(mat)
            result._ref_chain.add(mat)
        
        return result
    
    @classmethod
    def empty(cls, rows: int, cols: int, nnz: int, dtype: str = 'float64') -> 'SclCSC':
        """Create empty matrix with pre-allocated arrays."""
        validate_dtype(dtype)
        
        data = empty(nnz, dtype=dtype)
        indices = empty(nnz, dtype='int64')
        indptr = zeros(cols + 1, dtype='int64')
        
        return cls(data, indices, indptr, shape=(rows, cols), ownership=Ownership.OWNED)
    
    @classmethod
    def zeros(cls, rows: int, cols: int, dtype: str = 'float64') -> 'SclCSC':
        """Create zero matrix."""
        return cls.empty(rows, cols, 0, dtype)
    
    # =========================================================================
    # Virtual Backend Operations
    # =========================================================================
    
    @classmethod
    def _create_virtual(
        cls,
        sources: List['SclCSC'],
        indices_list: Optional[List[Optional[Array]]] = None
    ) -> 'SclCSC':
        """Create virtual matrix from sources (for hstack)."""
        if len(sources) == 0:
            return cls.zeros(0, 0)
        
        rows = sources[0].rows
        dtype = sources[0].dtype
        
        for src in sources[1:]:
            if src.rows != rows:
                raise ValueError(f"Row mismatch: {rows} vs {src.rows}")
        
        chunks = []
        total_cols = 0
        ref_chain = RefChain()
        
        for i, src in enumerate(sources):
            local_indices = indices_list[i] if indices_list else None
            length = len(local_indices) if local_indices is not None else src.cols
            
            chunks.append(ChunkInfo(
                source=src,
                local_indices=local_indices,
                offset=total_cols,
                length=length
            ))
            
            total_cols += length
            ref_chain.add(src)
        
        storage = VirtualStorage(
            chunks=chunks,
            total_primary=total_cols,
            secondary_size=rows,
            dtype=dtype
        ).flatten()
        
        result = cls(
            _storage=storage,
            shape=(rows, total_cols),
            backend=Backend.VIRTUAL,
            _ref_chain=ref_chain
        )
        result._shape = (rows, total_cols)
        result._dtype = dtype
        result._nnz = -1
        
        return result
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_scipy(self) -> Any:
        """Convert to scipy.sparse.csc_matrix."""
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
        
        return sp.csc_matrix((data_np, indices_np, indptr_np), shape=self.shape)
    
    def to_owned(self) -> 'SclCSC':
        """Convert to owned Custom backend."""
        if self._backend == Backend.CUSTOM and self.is_owned:
            return self.copy()
        
        self._ensure_custom()
        return self.copy()
    
    def tocsr(self) -> 'SclCSR':
        """Convert to CSR format."""
        from ._csr import SclCSR
        
        scipy_csc = self.to_scipy()
        scipy_csr = scipy_csc.tocsr()
        return SclCSR.from_scipy(scipy_csr, copy=True)
    
    def tocsc(self) -> 'SclCSC':
        """Return self (already CSC)."""
        return self
    
    def to_csc(self) -> 'SclCSC':
        """Return self (alias for tocsc)."""
        return self
    
    def to_csr(self) -> 'SclCSR':
        """Convert to CSR format (alias for tocsr)."""
        return self.tocsr()
    
    def copy(self) -> 'SclCSC':
        """Create deep copy."""
        self._ensure_custom()
        
        return SclCSC(
            data=self._storage.data.copy(),
            indices=self._storage.indices.copy(),
            indptr=self._storage.indptr.copy(),
            shape=self.shape,
            col_lengths=self._storage.primary_lengths.copy() if self._storage.primary_lengths else None,
            ownership=Ownership.OWNED
        )
    
    def _ensure_custom(self) -> None:
        """Ensure backend is Custom."""
        if self._backend == Backend.CUSTOM:
            return
        
        if self._backend == Backend.VIRTUAL:
            self._materialize_virtual()
        elif self._backend == Backend.MAPPED:
            self._load_mapped()
    
    def materialize(self) -> 'SclCSC':
        """Materialize the matrix (ensure all data is in memory).
        
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
        
        Returns:
            Sum value(s).
        """
        self._ensure_custom()
        
        if axis is None:
            total = 0.0
            for k in range(self.nnz):
                total += self._storage.data[k]
            return total
        
        elif axis == 0:  # Column sums (efficient for CSC)
            result = zeros(self.cols, dtype='float64')
            for j in range(self.cols):
                start = self._storage.indptr[j]
                end = self._storage.indptr[j + 1]
                col_sum = 0.0
                for k in range(start, end):
                    col_sum += self._storage.data[k]
                result[j] = col_sum
            return result
        
        else:  # Row sums (axis=1)
            result = zeros(self.rows, dtype='float64')
            for k in range(self.nnz):
                row_idx = self._storage.indices[k]
                result[row_idx] += self._storage.data[k]
            return result
    
    def mean(self, axis: Optional[int] = None) -> Union[float, 'Array']:
        """Compute the mean of matrix elements.
        
        Args:
            axis: Axis along which to compute mean.
        
        Returns:
            Mean value(s).
        """
        self._ensure_custom()
        
        if axis is None:
            n_total = self.rows * self.cols
            return self.sum() / n_total if n_total > 0 else 0.0
        
        elif axis == 0:  # Column means
            sums = self.sum(axis=0)
            result = zeros(self.cols, dtype='float64')
            for j in range(self.cols):
                result[j] = sums[j] / self.rows if self.rows > 0 else 0.0
            return result
        
        else:  # Row means (axis=1)
            sums = self.sum(axis=1)
            result = zeros(self.rows, dtype='float64')
            for i in range(self.rows):
                result[i] = sums[i] / self.cols if self.cols > 0 else 0.0
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
            min_val = 0.0
            for k in range(self.nnz):
                val = self._storage.data[k]
                if val < min_val:
                    min_val = val
            if self.nnz < self.rows * self.cols:
                min_val = min(min_val, 0.0)
            return min_val
        
        elif axis == 0:  # Column minimums
            result = zeros(self.cols, dtype='float64')
            for j in range(self.cols):
                start = self._storage.indptr[j]
                end = self._storage.indptr[j + 1]
                nnz_col = end - start
                
                if nnz_col == 0:
                    result[j] = 0.0
                elif nnz_col < self.rows:
                    col_min = 0.0
                    for k in range(start, end):
                        val = self._storage.data[k]
                        if val < col_min:
                            col_min = val
                    result[j] = col_min
                else:
                    col_min = self._storage.data[start]
                    for k in range(start + 1, end):
                        val = self._storage.data[k]
                        if val < col_min:
                            col_min = val
                    result[j] = col_min
            return result
        
        else:  # Row minimums (axis=1)
            result = zeros(self.rows, dtype='float64')
            import math
            for i in range(self.rows):
                result[i] = math.inf
            
            row_nnz = zeros(self.rows, dtype='int64')
            
            for k in range(self.nnz):
                row_idx = self._storage.indices[k]
                val = self._storage.data[k]
                row_nnz[row_idx] += 1
                if val < result[row_idx]:
                    result[row_idx] = val
            
            for i in range(self.rows):
                if row_nnz[i] < self.cols:
                    result[i] = min(result[i], 0.0)
                elif row_nnz[i] == 0:
                    result[i] = 0.0
            
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
            max_val = 0.0
            for k in range(self.nnz):
                val = self._storage.data[k]
                if val > max_val:
                    max_val = val
            if self.nnz < self.rows * self.cols:
                max_val = max(max_val, 0.0)
            return max_val
        
        elif axis == 0:  # Column maximums
            result = zeros(self.cols, dtype='float64')
            for j in range(self.cols):
                start = self._storage.indptr[j]
                end = self._storage.indptr[j + 1]
                nnz_col = end - start
                
                if nnz_col == 0:
                    result[j] = 0.0
                elif nnz_col < self.rows:
                    col_max = 0.0
                    for k in range(start, end):
                        val = self._storage.data[k]
                        if val > col_max:
                            col_max = val
                    result[j] = col_max
                else:
                    col_max = self._storage.data[start]
                    for k in range(start + 1, end):
                        val = self._storage.data[k]
                        if val > col_max:
                            col_max = val
                    result[j] = col_max
            return result
        
        else:  # Row maximums (axis=1)
            result = zeros(self.rows, dtype='float64')
            import math
            for i in range(self.rows):
                result[i] = -math.inf
            
            row_nnz = zeros(self.rows, dtype='int64')
            
            for k in range(self.nnz):
                row_idx = self._storage.indices[k]
                val = self._storage.data[k]
                row_nnz[row_idx] += 1
                if val > result[row_idx]:
                    result[row_idx] = val
            
            for i in range(self.rows):
                if row_nnz[i] < self.cols:
                    result[i] = max(result[i], 0.0)
                elif row_nnz[i] == 0:
                    result[i] = 0.0
            
            return result
    
    def _materialize_virtual(self) -> None:
        """Materialize Virtual backend to Custom."""
        if not isinstance(self._storage, VirtualStorage):
            return
        
        vstorage = self._storage
        total_nnz = vstorage.nnz
        
        new_data = empty(total_nnz, dtype=self.dtype)
        new_indices = empty(total_nnz, dtype='int64')
        new_indptr = zeros(vstorage.total_primary + 1, dtype='int64')
        
        data_pos = 0
        col_pos = 0
        
        for chunk in vstorage.chunks:
            src = chunk.source
            
            if hasattr(src, '_ensure_custom'):
                src._ensure_custom()
            
            if chunk.is_identity:
                src_nnz = src.nnz
                for k in range(src_nnz):
                    new_data[data_pos + k] = src.data[k]
                    new_indices[data_pos + k] = src.indices[k]
                
                for j in range(src.cols):
                    new_indptr[col_pos + j + 1] = new_indptr[col_pos + j] + (src.indptr[j + 1] - src.indptr[j])
                
                data_pos += src_nnz
                col_pos += src.cols
            else:
                for local_j, src_j in enumerate(chunk.local_indices.tolist()):
                    start = src.indptr[src_j]
                    end = src.indptr[src_j + 1]
                    length = end - start
                    
                    for k in range(length):
                        new_data[data_pos] = src.data[start + k]
                        new_indices[data_pos] = src.indices[start + k]
                        data_pos += 1
                    
                    new_indptr[col_pos + 1] = new_indptr[col_pos] + length
                    col_pos += 1
        
        self._storage = CustomStorage(
            data=new_data,
            indices=new_indices,
            indptr=new_indptr,
            primary_lengths=self._compute_col_lengths(new_indptr, vstorage.total_primary),
            ownership=Ownership.OWNED
        )
        self._backend = Backend.CUSTOM
        self._ownership = OwnershipTracker.owned()
        self._nnz = total_nnz
    
    def _load_mapped(self) -> None:
        """Load Mapped backend."""
        raise NotImplementedError("Mapped backend loading not yet implemented")
    
    # =========================================================================
    # Slicing Operations
    # =========================================================================
    
    def slice_cols(
        self,
        col_indices: Union[List[int], Array],
        strategy: str = 'auto'
    ) -> 'SclCSC':
        """Extract subset of columns.
        
        Args:
            col_indices: Column indices to extract.
            strategy: 'auto', 'virtual', or 'copy'.
            
        Returns:
            SclCSC with selected columns.
        """
        if isinstance(col_indices, list):
            col_indices = from_list(col_indices, dtype='int64')
        
        n_select = len(col_indices)
        
        if strategy == 'auto':
            if n_select < self.cols * 0.5:
                strategy = 'virtual'
            else:
                strategy = 'copy'
        
        if strategy == 'virtual':
            return self._create_virtual([self], [col_indices])
        else:
            return self._copy_cols(col_indices)
    
    def _copy_cols(self, col_indices: Array) -> 'SclCSC':
        """Copy selected columns."""
        self._ensure_custom()
        
        new_cols = len(col_indices)
        
        total_nnz = 0
        for j in range(new_cols):
            col_idx = col_indices[j]
            total_nnz += self._storage.primary_lengths[col_idx]
        
        new_data = empty(total_nnz, dtype=self.dtype)
        new_indices = empty(total_nnz, dtype='int64')
        new_indptr = zeros(new_cols + 1, dtype='int64')
        
        pos = 0
        for j in range(new_cols):
            col_idx = col_indices[j]
            start = self._storage.indptr[col_idx]
            end = self._storage.indptr[col_idx + 1]
            
            for k in range(start, end):
                new_data[pos] = self._storage.data[k]
                new_indices[pos] = self._storage.indices[k]
                pos += 1
            
            new_indptr[j + 1] = pos
        
        return SclCSC(new_data, new_indices, new_indptr, shape=(self.rows, new_cols), ownership=Ownership.OWNED)
    
    def slice_rows(
        self,
        row_indices: Union[List[int], Array],
        lazy: bool = True
    ) -> 'SclCSC':
        """Extract subset of rows.
        
        Args:
            row_indices: Row indices to keep.
            lazy: If True, mark as view without copying.
            
        Returns:
            SclCSC with selected rows.
        """
        if isinstance(row_indices, list):
            row_indices = from_list(row_indices, dtype='int64')
        
        if lazy and self._row_mask is None:
            result = SclCSC(
                _storage=self._storage,
                shape=(len(row_indices), self.cols),
                backend=self._backend,
                _ref_chain=RefChain()
            )
            result._ref_chain.add(self)
            result._dtype = self.dtype
            result._nnz = -1
            
            result._row_mask = zeros(self.rows, dtype='uint8')
            result._row_mapping = zeros(self.rows, dtype='int64')
            for new_idx in range(len(row_indices)):
                old_idx = row_indices[new_idx]
                result._row_mask[old_idx] = 1
                result._row_mapping[old_idx] = new_idx
            
            result._ownership = OwnershipTracker.view(self)
            return result
        else:
            return self._copy_with_row_filter(row_indices)
    
    def _copy_with_row_filter(self, row_indices: Array) -> 'SclCSC':
        """Copy with row filtering."""
        self._ensure_custom()
        
        new_rows = len(row_indices)
        
        row_mask = zeros(self.rows, dtype='uint8')
        row_mapping = zeros(self.rows, dtype='int64')
        for new_idx in range(new_rows):
            old_idx = row_indices[new_idx]
            row_mask[old_idx] = 1
            row_mapping[old_idx] = new_idx
        
        total_nnz = 0
        for k in range(self.nnz):
            if row_mask[self._storage.indices[k]] == 1:
                total_nnz += 1
        
        new_data = empty(total_nnz, dtype=self.dtype)
        new_indices = empty(total_nnz, dtype='int64')
        new_indptr = zeros(self.cols + 1, dtype='int64')
        
        pos = 0
        for j in range(self.cols):
            start = self._storage.indptr[j]
            end = self._storage.indptr[j + 1]
            
            for k in range(start, end):
                row_idx = self._storage.indices[k]
                if row_mask[row_idx] == 1:
                    new_data[pos] = self._storage.data[k]
                    new_indices[pos] = row_mapping[row_idx]
                    pos += 1
            
            new_indptr[j + 1] = pos
        
        return SclCSC(new_data, new_indices, new_indptr, shape=(new_rows, self.cols), ownership=Ownership.OWNED)
    
    def get_col(self, j: int) -> Tuple[Array, Array]:
        """Get column as sparse (indices, values)."""
        self._ensure_custom()
        
        if j < 0 or j >= self.cols:
            raise IndexError(f"Column {j} out of bounds [0, {self.cols})")
        
        start = self._storage.indptr[j]
        end = self._storage.indptr[j + 1]
        length = end - start
        
        if length == 0:
            return empty(0, dtype='int64'), empty(0, dtype=self.dtype)
        
        col_indices = empty(length, dtype='int64')
        col_values = empty(length, dtype=self.dtype)
        
        for k in range(length):
            col_indices[k] = self._storage.indices[start + k]
            col_values[k] = self._storage.data[start + k]
        
        return col_indices, col_values
    
    def get_col_dense(self, j: int) -> Array:
        """Get column as dense array."""
        indices, values = self.get_col(j)
        
        result = zeros(self.rows, dtype=self.dtype)
        for k in range(len(indices)):
            result[indices[k]] = values[k]
        
        return result
    
    # =========================================================================
    # Indexing
    # =========================================================================
    
    def __getitem__(self, key) -> Union['SclCSC', Array, float]:
        """Support mat[j], mat[i, j], mat[:, i:j], mat[:, [...]].
        
        For CSC, single index returns column.
        """
        if isinstance(key, int):
            return self.get_col_dense(key)
        
        if isinstance(key, slice):
            start, stop, step = key.indices(self.cols)
            if step == 1:
                indices = list(range(start, stop))
            else:
                indices = list(range(start, stop, step))
            return self.slice_cols(indices)
        
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            
            # mat[i, j]
            if isinstance(row_key, int) and isinstance(col_key, int):
                return self._get_element(row_key, col_key)
            
            # mat[:, j]
            if row_key == slice(None) and isinstance(col_key, int):
                return self.get_col_dense(col_key)
            
            # mat[i, :]
            if isinstance(row_key, int) and col_key == slice(None):
                return self._get_row_dense(row_key)
            
            # mat[:, i:j]
            if row_key == slice(None) and isinstance(col_key, slice):
                start, stop, step = col_key.indices(self.cols)
                if step == 1:
                    indices = list(range(start, stop))
                else:
                    indices = list(range(start, stop, step))
                return self.slice_cols(indices)
            
            # mat[:, [...]]
            if row_key == slice(None) and isinstance(col_key, (list, Array)):
                return self.slice_cols(col_key)
            
            # mat[[...], :]
            if isinstance(row_key, (list, Array)) and col_key == slice(None):
                return self.slice_rows(row_key)
            
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
        
        start = self._storage.indptr[j]
        end = self._storage.indptr[j + 1]
        
        for k in range(start, end):
            if self._storage.indices[k] == i:
                return self._storage.data[k]
        return 0.0
    
    def _get_row_dense(self, i: int) -> Array:
        """Get row as dense array."""
        self._ensure_custom()
        
        result = zeros(self.cols, dtype=self.dtype)
        for j in range(self.cols):
            start = self._storage.indptr[j]
            end = self._storage.indptr[j + 1]
            for k in range(start, end):
                if self._storage.indices[k] == i:
                    result[j] = self._storage.data[k]
                    break
        return result
    
    # =========================================================================
    # C API Interface
    # =========================================================================
    
    def get_c_pointers(self) -> Tuple:
        """Get C-compatible pointers."""
        self._ensure_custom()
        
        ptrs = self._storage.get_pointers()
        return ptrs + (self.rows, self.cols, self.nnz)
    
    # =========================================================================
    # Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return (
            f"SclCSC(shape={self.shape}, nnz={self.nnz}, "
            f"dtype={self.dtype}, backend={self.backend.value})"
        )
    
    def info(self) -> str:
        """Get detailed information string."""
        lines = [
            f"SclCSC Matrix:",
            f"  shape: {self.shape}",
            f"  nnz: {self.nnz}",
            f"  dtype: {self.dtype}",
            f"  backend: {self.backend.value}",
            f"  ownership: {self.ownership.value}",
            f"  is_view: {self.is_view}",
        ]
        if isinstance(self._storage, CustomStorage):
            lines.append(f"  memory: {self._storage.nbytes / 1024:.2f} KB")
        return '\n'.join(lines)


# Backward compatibility alias
CSC = SclCSC
