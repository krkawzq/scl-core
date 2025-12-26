"""
SCL Sparse - Intelligent Sparse Matrix

Zero external dependencies. Provides automatic backend management:
- Custom mode: All data in RAM (owned or borrowed)
- Virtual mode: Zero-copy views and compositions
- Mapped mode: Data on disk, loaded on demand

CSR and CSC have symmetric interfaces.

Scipy Compatibility (optional):
    mat.to_scipy()  # Convert to scipy.sparse when needed
"""

from __future__ import annotations

import ctypes
from ctypes import c_int64, c_uint8, c_double, c_int32, c_void_p, POINTER, byref, cast
from typing import Optional, Union, Tuple, Iterator, Sequence, Callable, TYPE_CHECKING
import weakref
import math

from scl._ffi import get_lib_with_signatures, check_error, SclError
from scl.array import RealArray, IndexArray, ByteArray, SIZEOF_REAL, SIZEOF_INDEX
from scl._dtypes import DType
from scl._backend import (
    Backend, Ownership, SourceType,
    StorageInfo, CustomStorage, VirtualStorage, MappedStorage, ChunkInfo,
    BackendFactory, estimate_memory, suggest_backend,
)
from scl._config import (
    config, MaterializeStrategy, MemoryStrategy, NormType,
    MaterializeConfig, SliceConfig,
)


# =============================================================================
# Lazy Level Control
# =============================================================================

class LazyLevel:
    """Controls lazy evaluation behavior."""
    EAGER = 0       # Materialize immediately
    DEFERRED = 1    # Defer until explicitly requested
    STREAMING = 2   # Stream from disk, never fully load


# =============================================================================
# Sparse Adapter Protocol
# =============================================================================

class SparseAdapter:
    """
    Protocol for custom sparse matrix adapters.

    Allows Python code to provide custom loading/processing logic
    that integrates with the backend system.

    Example:
        class MyAdapter(SparseAdapter):
            def load_row(self, i):
                return values, indices

            def shape(self):
                return (n_rows, n_cols)
    """

    def load_row(self, i: int) -> Tuple[Sequence[float], Sequence[int]]:
        """Load values and indices for row i."""
        raise NotImplementedError

    def load_col(self, j: int) -> Tuple[Sequence[float], Sequence[int]]:
        """Load values and indices for column j."""
        raise NotImplementedError

    def load_rows(self, indices: Sequence[int]) -> Tuple[RealArray, IndexArray, IndexArray]:
        """Load multiple rows, return (data, indices, indptr)."""
        raise NotImplementedError

    def load_cols(self, indices: Sequence[int]) -> Tuple[RealArray, IndexArray, IndexArray]:
        """Load multiple columns, return (data, indices, indptr)."""
        raise NotImplementedError

    def shape(self) -> Tuple[int, int]:
        """Return (rows, cols)."""
        raise NotImplementedError

    def nnz(self) -> int:
        """Return total number of non-zeros."""
        raise NotImplementedError


# =============================================================================
# Base Sparse Matrix
# =============================================================================

class SparseBase:
    """
    Base class for sparse matrices (CSR and CSC).

    Provides common interface and shared functionality with
    smart strategy support via property/setter pattern.
    """

    __slots__ = (
        "_storage",
        "_rows", "_cols", "_nnz",
        "_data", "_indices", "_indptr",
        "_adapter",
        "_config_overrides",
        "__weakref__",
    )

    # Tag for format identification (override in subclasses)
    _format = "base"
    _primary_axis = 0    # 0 for CSR (row-major), 1 for CSC (col-major)

    def __init__(self):
        self._storage: StorageInfo = CustomStorage()
        self._rows: int = 0
        self._cols: int = 0
        self._nnz: int = 0
        self._data: Optional[RealArray] = None
        self._indices: Optional[IndexArray] = None
        self._indptr: Optional[IndexArray] = None
        self._adapter: Optional[SparseAdapter] = None
        self._config_overrides: dict = {}

    def __del__(self):
        self._release()

    def _release(self):
        """Release C++ resources."""
        if isinstance(self._storage, MappedStorage):
            if self._storage.handle != 0:
                try:
                    lib = get_lib_with_signatures()
                    lib.scl_mmap_release(self._storage.handle)
                except Exception:
                    pass
                self._storage.handle = 0

    # -------------------------------------------------------------------------
    # Strategy Configuration (Property/Setter Pattern)
    # -------------------------------------------------------------------------

    @property
    def materialize_strategy(self) -> MaterializeStrategy:
        """Get materialize strategy for this matrix."""
        if "materialize_strategy" in self._config_overrides:
            return self._config_overrides["materialize_strategy"]
        return config.materialize.strategy

    @materialize_strategy.setter
    def materialize_strategy(self, value: MaterializeStrategy):
        """Set materialize strategy for this matrix."""
        self._config_overrides["materialize_strategy"] = value

    @property
    def memory_strategy(self) -> MemoryStrategy:
        """Get memory strategy for this matrix."""
        if "memory_strategy" in self._config_overrides:
            return self._config_overrides["memory_strategy"]
        return config.memory.strategy

    @memory_strategy.setter
    def memory_strategy(self, value: MemoryStrategy):
        """Set memory strategy for this matrix."""
        self._config_overrides["memory_strategy"] = value

    @property
    def lazy_slicing(self) -> bool:
        """Whether slicing creates lazy views."""
        if "lazy_slicing" in self._config_overrides:
            return self._config_overrides["lazy_slicing"]
        return config.slice.lazy

    @lazy_slicing.setter
    def lazy_slicing(self, value: bool):
        """Set lazy slicing behavior."""
        self._config_overrides["lazy_slicing"] = value

    def configure(self, **kwargs) -> "SparseBase":
        """
        Configure multiple options at once.

        Args:
            materialize_strategy: MaterializeStrategy
            memory_strategy: MemoryStrategy
            lazy_slicing: bool

        Returns:
            self (for chaining)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self._config_overrides[key] = value
        return self

    # -------------------------------------------------------------------------
    # Properties (Common Interface)
    # -------------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape (rows, cols)."""
        return (self._rows, self._cols)

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return self._nnz

    @property
    def ndim(self) -> int:
        """Number of dimensions (always 2)."""
        return 2

    @property
    def format(self) -> str:
        """Sparse format ('csr' or 'csc')."""
        return self._format

    @property
    def dtype(self) -> DType:
        """Data type of values."""
        return DType.FLOAT64

    @property
    def backend(self) -> Backend:
        """Current backend type."""
        return self._storage.backend

    @property
    def is_materialized(self) -> bool:
        """Whether data is loaded in memory."""
        if self._data is None:
            return False
        if isinstance(self._storage, VirtualStorage):
            return not self._storage.has_pending_ops()
        return True

    @property
    def is_view(self) -> bool:
        """Whether this is a view of another matrix."""
        return self._storage.is_view

    @property
    def is_contiguous(self) -> bool:
        """Whether data is contiguous in memory."""
        return self._storage.is_contiguous

    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        if self._data is not None:
            return self._data.nbytes + self._indices.nbytes + self._indptr.nbytes
        return estimate_memory(self._rows, self._nnz)

    # -------------------------------------------------------------------------
    # Data Access (Materializes if needed)
    # -------------------------------------------------------------------------

    @property
    def data(self) -> RealArray:
        """Non-zero values."""
        self.materialize()
        return self._data

    @property
    def indices(self) -> IndexArray:
        """Column indices (CSR) or row indices (CSC)."""
        self.materialize()
        return self._indices

    @property
    def indptr(self) -> IndexArray:
        """Row pointers (CSR) or column pointers (CSC)."""
        self.materialize()
        return self._indptr

    # -------------------------------------------------------------------------
    # Element Access
    # -------------------------------------------------------------------------

    def _get_element(self, i: int, j: int) -> float:
        """Get single element value (implemented by subclass)."""
        raise NotImplementedError

    def __getitem__(self, key):
        """
        Slice operation (lazy or eager based on configuration).

        Supports:
        - mat[row_mask]: Row selection
        - mat[row_mask, col_mask]: Row and column selection
        - mat[:, col_mask]: Column selection
        - mat[i, j]: Single element (returns scalar)
        """
        raise NotImplementedError  # Implemented by subclasses

    # -------------------------------------------------------------------------
    # Materialization
    # -------------------------------------------------------------------------

    def materialize(self, force: bool = False) -> "SparseBase":
        """
        Materialize: Execute all lazy operations, load to memory.

        Args:
            force: Force re-materialization even if already materialized

        Returns:
            self (for chaining)
        """
        if self.is_materialized and not force:
            return self

        # Handle adapter-based loading
        if self._adapter is not None:
            self._materialize_from_adapter()
            return self

        # Handle different backends
        if isinstance(self._storage, MappedStorage):
            self._materialize_from_mapped()
        elif isinstance(self._storage, VirtualStorage):
            self._materialize_from_virtual()
        # CustomStorage is already in memory

        return self

    def _primary_size(self) -> int:
        """Return primary dimension size."""
        raise NotImplementedError

    def _materialize_from_adapter(self):
        """Materialize from Python adapter (subclass implements)."""
        raise NotImplementedError

    def _materialize_from_mapped(self):
        """Materialize from memory-mapped file (subclass implements)."""
        raise NotImplementedError

    def _materialize_from_virtual(self):
        """Materialize from virtual storage (subclass implements)."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def iter_primary(self) -> Iterator[Tuple[RealArray, IndexArray]]:
        """
        Iterate over primary axis (rows for CSR, columns for CSC).

        Yields:
            (values, indices) for each row/column
        """
        raise NotImplementedError

    def iter_rows(self) -> Iterator[Tuple[int, RealArray, IndexArray]]:
        """
        Iterate over rows.

        Yields:
            (row_idx, values, col_indices) for each row
        """
        raise NotImplementedError

    def iter_cols(self) -> Iterator[Tuple[int, RealArray, IndexArray]]:
        """
        Iterate over columns.

        Yields:
            (col_idx, values, row_indices) for each column
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def sum(self, axis: Optional[int] = None) -> Union[float, RealArray]:
        """
        Sum of elements.

        Args:
            axis: None=global, 0=column, 1=row
        """
        raise NotImplementedError

    def mean(self, axis: Optional[int] = None) -> Union[float, RealArray]:
        """Mean of elements."""
        raise NotImplementedError

    def min(self, axis: Optional[int] = None) -> Union[float, RealArray]:
        """Minimum value."""
        raise NotImplementedError

    def max(self, axis: Optional[int] = None) -> Union[float, RealArray]:
        """Maximum value."""
        raise NotImplementedError

    def nonzero(self) -> Tuple[IndexArray, IndexArray]:
        """Return indices of non-zero elements."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Conversion
    # -------------------------------------------------------------------------

    def copy(self) -> "SparseBase":
        """Create a deep copy."""
        raise NotImplementedError

    def to_scipy(self):
        """Convert to scipy.sparse matrix."""
        raise NotImplementedError

    def to_dense(self) -> RealArray:
        """Convert to dense (row-major) array."""
        raise NotImplementedError

    def astype(self, dtype: DType) -> "SparseBase":
        """Convert to different dtype."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------

    def __neg__(self) -> "SparseBase":
        """Negation: -A"""
        raise NotImplementedError

    def __add__(self, other) -> "SparseBase":
        """Addition: A + B"""
        raise NotImplementedError

    def __sub__(self, other) -> "SparseBase":
        """Subtraction: A - B"""
        raise NotImplementedError

    def __mul__(self, other) -> "SparseBase":
        """Element-wise multiplication: A * B or A * scalar"""
        raise NotImplementedError

    def __truediv__(self, other) -> "SparseBase":
        """Element-wise division: A / scalar"""
        raise NotImplementedError

    def __matmul__(self, other):
        """Matrix multiplication: A @ B"""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self):
        status = "materialized" if self.is_materialized else "lazy"
        backend_name = self._storage.backend.name.lower()
        return f"{type(self).__name__}(shape={self.shape}, nnz={self._nnz}, {status}, {backend_name})"

    def __len__(self):
        return self._rows

    def info(self) -> dict:
        """Get detailed information about the matrix."""
        return {
            "type": type(self).__name__,
            "format": self._format,
            "shape": self.shape,
            "nnz": self._nnz,
            "dtype": self.dtype.name,
            "backend": self._storage.backend.name,
            "ownership": self._storage.ownership.name,
            "is_materialized": self.is_materialized,
            "is_view": self.is_view,
            "memory_bytes": self.memory_bytes,
            "density": self._nnz / (self._rows * self._cols) if self._rows * self._cols > 0 else 0,
        }


# =============================================================================
# SclCSR - CSR Sparse Matrix
# =============================================================================

class SclCSR(SparseBase):
    """
    Compressed Sparse Row (CSR) Matrix.

    Row-major storage: efficient for row operations.

    Features:
    - Lazy slicing with configurable strategy
    - Multiple backend support (custom, virtual, mapped)
    - Property-based strategy configuration
    - Zero-copy views when possible
    """

    _format = "csr"
    _primary_axis = 0

    def _primary_size(self) -> int:
        return self._rows

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_arrays(cls, data: Union[RealArray, Sequence[float]],
                    indices: Union[IndexArray, Sequence[int]],
                    indptr: Union[IndexArray, Sequence[int]],
                    shape: Tuple[int, int],
                    copy: bool = False) -> "SclCSR":
        """
        Create from raw arrays.

        Args:
            data: Non-zero values
            indices: Column indices
            indptr: Row pointers
            shape: (rows, cols)
            copy: Whether to copy input arrays
        """
        obj = cls()
        obj._rows, obj._cols = shape

        # Convert to SCL arrays if needed
        if isinstance(data, RealArray):
            obj._data = data.copy() if copy else data
        else:
            obj._data = RealArray.from_sequence(data)

        if isinstance(indices, IndexArray):
            obj._indices = indices.copy() if copy else indices
        else:
            obj._indices = IndexArray.from_sequence(indices)

        if isinstance(indptr, IndexArray):
            obj._indptr = indptr.copy() if copy else indptr
        else:
            obj._indptr = IndexArray.from_sequence(indptr)

        obj._nnz = obj._data.size
        obj._storage = CustomStorage(
            ownership=Ownership.OWNED,
            memory_bytes=obj._data.nbytes + obj._indices.nbytes + obj._indptr.nbytes,
        )

        return obj

    @classmethod
    def from_scipy(cls, sp_matrix) -> "SclCSR":
        """
        Create from scipy.sparse matrix.

        Args:
            sp_matrix: scipy.sparse matrix (converts to CSR)
        """
        from scipy import sparse

        if not sparse.isspmatrix_csr(sp_matrix):
            sp_matrix = sp_matrix.tocsr()

        from scl.array import from_numpy
        import numpy as np

        data = from_numpy(np.ascontiguousarray(sp_matrix.data, dtype=np.float64))
        indices = from_numpy(np.ascontiguousarray(sp_matrix.indices, dtype=np.int64))
        indptr = from_numpy(np.ascontiguousarray(sp_matrix.indptr, dtype=np.int64))

        return cls.from_arrays(data, indices, indptr, sp_matrix.shape)

    @classmethod
    def from_file(cls, filepath: str, max_pages: int = 64) -> "SclCSR":
        """
        Create from SCL binary file (mapped mode).

        Args:
            filepath: Path to SCL binary file
            max_pages: Maximum resident pages
        """
        lib = get_lib_with_signatures()

        obj = cls()
        handle = c_int64()
        check_error(lib.scl_mmap_open_csr_file(
            filepath.encode("utf-8"),
            max_pages,
            byref(handle)
        ))

        # Get dimensions
        rows, cols, nnz = c_int64(), c_int64(), c_int64()
        check_error(lib.scl_mmap_csr_shape(
            handle.value, byref(rows), byref(cols), byref(nnz)
        ))
        obj._rows = rows.value
        obj._cols = cols.value
        obj._nnz = nnz.value

        obj._storage = MappedStorage(
            file_path=filepath,
            handle=handle.value,
            max_pages=max_pages,
        )

        return obj

    @classmethod
    def from_adapter(cls, adapter: SparseAdapter) -> "SclCSR":
        """
        Create from Python adapter.

        Args:
            adapter: Custom SparseAdapter implementation
        """
        obj = cls()
        obj._adapter = adapter
        obj._rows, obj._cols = adapter.shape()
        obj._nnz = adapter.nnz()
        obj._storage = VirtualStorage(source_type=SourceType.PARENT)

        return obj

    @classmethod
    def empty(cls, shape: Tuple[int, int]) -> "SclCSR":
        """Create empty matrix with given shape."""
        rows, cols = shape
        data = RealArray(0)
        indices = IndexArray(0)
        indptr = IndexArray(rows + 1)
        indptr.zero()
        return cls.from_arrays(data, indices, indptr, shape)

    @classmethod
    def eye(cls, n: int, m: Optional[int] = None, k: int = 0) -> "SclCSR":
        """
        Create identity matrix.

        Args:
            n: Number of rows
            m: Number of columns (default: n)
            k: Diagonal offset (default: 0)
        """
        m = m if m is not None else n
        diag_len = min(n, m)

        data = []
        indices = []
        indptr = [0]

        for i in range(n):
            j = i + k
            if 0 <= j < m:
                data.append(1.0)
                indices.append(j)
            indptr.append(len(data))

        return cls.from_arrays(data, indices, indptr, (n, m))

    # -------------------------------------------------------------------------
    # Element Access
    # -------------------------------------------------------------------------

    def _get_element(self, i: int, j: int) -> float:
        """Get single element value."""
        self.materialize()

        start = self._indptr[i]
        end = self._indptr[i + 1]
        for k in range(start, end):
            if self._indices[k] == j:
                return self._data[k]
        return 0.0

    def __getitem__(self, key):
        """Slice operation."""
        if isinstance(key, tuple):
            row_key, col_key = key if len(key) == 2 else (key[0], None)
        else:
            row_key = key
            col_key = None

        # Handle single element access
        if isinstance(row_key, int) and isinstance(col_key, int):
            return self._get_element(row_key, col_key)

        # Check if lazy slicing is enabled
        if not self.lazy_slicing:
            return self._slice_eager(row_key, col_key)

        return self._slice_lazy(row_key, col_key)

    def _slice_lazy(self, row_key, col_key) -> "SclCSR":
        """Create lazy view."""
        # Handle single row -> single row matrix
        if isinstance(row_key, int):
            row_key = [row_key]

        # Create virtual storage
        view = SclCSR()
        view._storage = VirtualStorage(source_type=SourceType.SLICE)
        view._storage.add_chunk(self, (0, self._rows), (0, self._cols))
        view._adapter = None

        # Copy config overrides
        view._config_overrides = self._config_overrides.copy()

        # Process row selection
        if row_key is not None and not (isinstance(row_key, slice) and row_key == slice(None)):
            view._storage.row_mask = self._to_mask(row_key, self._rows)

        # Process column selection
        if col_key is not None and not (isinstance(col_key, slice) and col_key == slice(None)):
            view._storage.col_mask = self._to_mask(col_key, self._cols)

        # Compute new dimensions
        row_mask = view._storage.row_mask
        col_mask = view._storage.col_mask
        view._rows = row_mask.count_nonzero() if row_mask else self._rows
        view._cols = col_mask.count_nonzero() if col_mask else self._cols

        # Estimate nnz (actual nnz computed on materialize)
        view._nnz = self._nnz  # Approximate

        return view

    def _slice_eager(self, row_key, col_key) -> "SclCSR":
        """Create eager copy of slice."""
        self.materialize()

        # Handle single row
        if isinstance(row_key, int):
            row_key = [row_key]

        # Convert to masks
        row_mask = None if row_key is None or (isinstance(row_key, slice) and row_key == slice(None)) else self._to_mask(row_key, self._rows)
        col_mask = None if col_key is None or (isinstance(col_key, slice) and col_key == slice(None)) else self._to_mask(col_key, self._cols)

        return self._extract_submatrix(row_mask, col_mask)

    def _to_mask(self, key, length: int) -> ByteArray:
        """Convert key to uint8 mask."""
        mask = ByteArray(length)

        if isinstance(key, slice):
            for i in range(*key.indices(length)):
                mask[i] = 1
        elif isinstance(key, (list, tuple)):
            for i in key:
                if isinstance(i, bool):
                    if i:
                        idx = list(key).index(i)
                        mask[idx] = 1
                else:
                    mask[i] = 1
        elif hasattr(key, '__iter__'):
            for i, v in enumerate(key):
                if isinstance(v, bool):
                    if v:
                        mask[i] = 1
                else:
                    mask[v] = 1
        else:
            mask[int(key)] = 1

        return mask

    def _extract_submatrix(self, row_mask: Optional[ByteArray],
                           col_mask: Optional[ByteArray]) -> "SclCSR":
        """Extract submatrix based on masks."""
        row_indices = [i for i in range(self._rows) if row_mask is None or row_mask[i]]
        col_set = None
        col_remap = None

        if col_mask is not None:
            col_set = set(j for j in range(self._cols) if col_mask[j])
            col_remap = {old: new for new, old in enumerate(sorted(col_set))}

        new_data = []
        new_indices = []
        new_indptr = [0]

        for i in row_indices:
            start = self._indptr[i]
            end = self._indptr[i + 1]
            for k in range(start, end):
                col = self._indices[k]
                if col_set is None or col in col_set:
                    new_data.append(self._data[k])
                    if col_remap:
                        new_indices.append(col_remap[col])
                    else:
                        new_indices.append(col)
            new_indptr.append(len(new_data))

        new_rows = len(row_indices)
        new_cols = len(col_set) if col_set else self._cols

        return SclCSR.from_arrays(new_data, new_indices, new_indptr, (new_rows, new_cols))

    # -------------------------------------------------------------------------
    # Row Access
    # -------------------------------------------------------------------------

    def getrow(self, i: int) -> Tuple[RealArray, IndexArray]:
        """
        Get row data efficiently.

        Args:
            i: Row index

        Returns:
            (values, column_indices)
        """
        self.materialize()

        if i < 0:
            i += self._rows
        if i < 0 or i >= self._rows:
            raise IndexError(f"Row index {i} out of range [0, {self._rows})")

        start = self._indptr[i]
        end = self._indptr[i + 1]
        length = end - start

        if length == 0:
            return RealArray(0), IndexArray(0)

        vals = RealArray(length)
        cols = IndexArray(length)
        for k in range(length):
            vals[k] = self._data[start + k]
            cols[k] = self._indices[start + k]

        return vals, cols

    def row_length(self, i: int) -> int:
        """Get number of non-zeros in row."""
        self.materialize()
        return self._indptr[i + 1] - self._indptr[i]

    # -------------------------------------------------------------------------
    # Materialization
    # -------------------------------------------------------------------------

    def _materialize_from_adapter(self):
        """Materialize from Python adapter."""
        storage = self._storage
        if isinstance(storage, VirtualStorage):
            row_mask = storage.row_mask
            col_mask = storage.col_mask
        else:
            row_mask = None
            col_mask = None

        if row_mask is not None:
            row_indices = [i for i in range(self._rows) if row_mask[i]]
        else:
            row_indices = list(range(self._rows))

        data, indices, indptr = self._adapter.load_rows(row_indices)

        if col_mask is not None:
            col_set = set(i for i in range(self._cols) if col_mask[i])
            col_remap = {old: new for new, old in enumerate(sorted(col_set))}

            new_data = []
            new_indices = []
            new_indptr = [0]

            for i in range(len(row_indices)):
                start = indptr[i]
                end = indptr[i + 1]
                for k in range(start, end):
                    col = indices[k]
                    if col in col_set:
                        new_data.append(data[k])
                        new_indices.append(col_remap[col])
                new_indptr.append(len(new_data))

            self._data = RealArray.from_sequence(new_data)
            self._indices = IndexArray.from_sequence(new_indices)
            self._indptr = IndexArray.from_sequence(new_indptr)
        else:
            self._data = data if isinstance(data, RealArray) else RealArray.from_sequence(data)
            self._indices = indices if isinstance(indices, IndexArray) else IndexArray.from_sequence(indices)
            self._indptr = indptr if isinstance(indptr, IndexArray) else IndexArray.from_sequence(indptr)

        self._rows = len(row_indices)
        self._cols = col_mask.count_nonzero() if col_mask else self._cols
        self._nnz = self._data.size

        # Update storage
        self._storage = CustomStorage(ownership=Ownership.OWNED)
        self._adapter = None

    def _materialize_from_mapped(self):
        """Materialize from memory-mapped file."""
        storage = self._storage
        if not isinstance(storage, MappedStorage):
            return

        lib = get_lib_with_signatures()
        handle = storage.handle

        self._data = RealArray(self._nnz)
        self._indices = IndexArray(self._nnz)
        self._indptr = IndexArray(self._rows + 1)

        check_error(lib.scl_mmap_csr_load_full(
            handle,
            c_void_p(self._data.data_ptr),
            c_void_p(self._indices.data_ptr),
            c_void_p(self._indptr.data_ptr)
        ))

    def _materialize_from_virtual(self):
        """Materialize from virtual storage."""
        storage = self._storage
        if not isinstance(storage, VirtualStorage):
            return

        # Get source matrix
        if not storage.chunks:
            return

        chunk = storage.chunks[0]
        source = chunk.get_source()
        if source is None:
            raise RuntimeError("Source matrix has been garbage collected")

        # Materialize source first
        source.materialize()

        # Apply masks
        row_mask = storage.row_mask
        col_mask = storage.col_mask

        if row_mask is None and col_mask is None:
            # Simple copy
            self._data = source._data.copy()
            self._indices = source._indices.copy()
            self._indptr = source._indptr.copy()
            self._nnz = source._nnz
        else:
            # Extract submatrix
            result = source._extract_submatrix(row_mask, col_mask)
            self._data = result._data
            self._indices = result._indices
            self._indptr = result._indptr
            self._rows = result._rows
            self._cols = result._cols
            self._nnz = result._nnz

        # Update storage
        self._storage = CustomStorage(ownership=Ownership.OWNED)

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def iter_rows(self) -> Iterator[Tuple[int, RealArray, IndexArray]]:
        """Iterate over rows."""
        self.materialize()
        for i in range(self._rows):
            vals, cols = self.getrow(i)
            yield i, vals, cols

    def iter_primary(self) -> Iterator[Tuple[RealArray, IndexArray]]:
        """Iterate over primary axis (rows)."""
        self.materialize()
        for i in range(self._rows):
            yield self.getrow(i)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def sum(self, axis: Optional[int] = None) -> Union[float, RealArray]:
        """Sum of elements."""
        self.materialize()

        if axis is None:
            total = 0.0
            for v in self._data:
                total += v
            return total

        elif axis == 1:  # Row sums
            result = RealArray(self._rows)
            for i in range(self._rows):
                start = self._indptr[i]
                end = self._indptr[i + 1]
                row_sum = 0.0
                for k in range(start, end):
                    row_sum += self._data[k]
                result[i] = row_sum
            return result

        elif axis == 0:  # Column sums
            result = RealArray(self._cols)
            result.zero()
            for i in range(self._rows):
                start = self._indptr[i]
                end = self._indptr[i + 1]
                for k in range(start, end):
                    result[self._indices[k]] += self._data[k]
            return result

        raise ValueError(f"Invalid axis: {axis}")

    def mean(self, axis: Optional[int] = None) -> Union[float, RealArray]:
        """Mean of elements."""
        sums = self.sum(axis)
        if axis is None:
            return sums / (self._rows * self._cols)
        elif axis == 1:
            for i in range(sums.size):
                sums[i] /= self._cols
            return sums
        else:
            for i in range(sums.size):
                sums[i] /= self._rows
            return sums

    def min(self, axis: Optional[int] = None) -> Union[float, RealArray]:
        """Minimum value."""
        self.materialize()

        if axis is None:
            min_val = float('inf')
            for v in self._data:
                if v < min_val:
                    min_val = v
            # Check if there are zeros
            if self._nnz < self._rows * self._cols and min_val > 0:
                min_val = 0.0
            return min_val

        elif axis == 1:  # Row min
            result = RealArray(self._rows)
            for i in range(self._rows):
                start = self._indptr[i]
                end = self._indptr[i + 1]
                if start == end:
                    result[i] = 0.0
                else:
                    row_min = float('inf')
                    for k in range(start, end):
                        if self._data[k] < row_min:
                            row_min = self._data[k]
                    # Check for implicit zeros
                    if (end - start) < self._cols and row_min > 0:
                        row_min = 0.0
                    result[i] = row_min
            return result

        elif axis == 0:  # Column min
            result = RealArray(self._cols)
            counts = [0] * self._cols
            for i in range(self._cols):
                result[i] = float('inf')

            for i in range(self._rows):
                start = self._indptr[i]
                end = self._indptr[i + 1]
                for k in range(start, end):
                    j = self._indices[k]
                    if self._data[k] < result[j]:
                        result[j] = self._data[k]
                    counts[j] += 1

            for j in range(self._cols):
                if counts[j] < self._rows and result[j] > 0:
                    result[j] = 0.0
                elif counts[j] == 0:
                    result[j] = 0.0

            return result

        raise ValueError(f"Invalid axis: {axis}")

    def max(self, axis: Optional[int] = None) -> Union[float, RealArray]:
        """Maximum value."""
        self.materialize()

        if axis is None:
            max_val = float('-inf')
            for v in self._data:
                if v > max_val:
                    max_val = v
            if self._nnz < self._rows * self._cols and max_val < 0:
                max_val = 0.0
            return max_val

        elif axis == 1:  # Row max
            result = RealArray(self._rows)
            for i in range(self._rows):
                start = self._indptr[i]
                end = self._indptr[i + 1]
                if start == end:
                    result[i] = 0.0
                else:
                    row_max = float('-inf')
                    for k in range(start, end):
                        if self._data[k] > row_max:
                            row_max = self._data[k]
                    if (end - start) < self._cols and row_max < 0:
                        row_max = 0.0
                    result[i] = row_max
            return result

        elif axis == 0:  # Column max
            result = RealArray(self._cols)
            counts = [0] * self._cols
            for i in range(self._cols):
                result[i] = float('-inf')

            for i in range(self._rows):
                start = self._indptr[i]
                end = self._indptr[i + 1]
                for k in range(start, end):
                    j = self._indices[k]
                    if self._data[k] > result[j]:
                        result[j] = self._data[k]
                    counts[j] += 1

            for j in range(self._cols):
                if counts[j] < self._rows and result[j] < 0:
                    result[j] = 0.0
                elif counts[j] == 0:
                    result[j] = 0.0

            return result

        raise ValueError(f"Invalid axis: {axis}")

    def nonzero(self) -> Tuple[IndexArray, IndexArray]:
        """Return indices of non-zero elements."""
        self.materialize()

        rows = IndexArray(self._nnz)
        cols = IndexArray(self._nnz)

        idx = 0
        for i in range(self._rows):
            start = self._indptr[i]
            end = self._indptr[i + 1]
            for k in range(start, end):
                rows[idx] = i
                cols[idx] = self._indices[k]
                idx += 1

        return rows, cols

    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------

    def __neg__(self) -> "SclCSR":
        """Negation: -A"""
        self.materialize()
        new_data = RealArray(self._nnz)
        for k in range(self._nnz):
            new_data[k] = -self._data[k]
        return SclCSR.from_arrays(new_data, self._indices.copy(), self._indptr.copy(), self.shape)

    def __mul__(self, other) -> "SclCSR":
        """Element-wise multiplication with scalar."""
        if isinstance(other, (int, float)):
            self.materialize()
            new_data = RealArray(self._nnz)
            for k in range(self._nnz):
                new_data[k] = self._data[k] * other
            return SclCSR.from_arrays(new_data, self._indices.copy(), self._indptr.copy(), self.shape)
        raise NotImplementedError("Element-wise matrix multiplication not implemented")

    def __rmul__(self, other) -> "SclCSR":
        """Right multiplication with scalar."""
        return self.__mul__(other)

    def __truediv__(self, other) -> "SclCSR":
        """Element-wise division by scalar."""
        if isinstance(other, (int, float)):
            return self.__mul__(1.0 / other)
        raise NotImplementedError("Element-wise matrix division not implemented")

    # -------------------------------------------------------------------------
    # Conversion
    # -------------------------------------------------------------------------

    def copy(self) -> "SclCSR":
        """Create a deep copy."""
        self.materialize()
        return SclCSR.from_arrays(
            self._data.copy(),
            self._indices.copy(),
            self._indptr.copy(),
            self.shape
        )

    def to_scipy(self):
        """Convert to scipy.sparse.csr_matrix."""
        self.materialize()
        from scipy import sparse
        from scl.array import to_numpy

        return sparse.csr_matrix(
            (to_numpy(self._data), to_numpy(self._indices), to_numpy(self._indptr)),
            shape=self.shape
        )

    def to_dense(self) -> RealArray:
        """Convert to dense (row-major) array."""
        self.materialize()
        dense = RealArray(self._rows * self._cols)
        dense.zero()

        for i in range(self._rows):
            start = self._indptr[i]
            end = self._indptr[i + 1]
            for k in range(start, end):
                dense[i * self._cols + self._indices[k]] = self._data[k]

        return dense

    def to_csc(self) -> "SclCSC":
        """Convert to CSC format."""
        self.materialize()

        # Count elements per column
        col_counts = [0] * self._cols
        for k in range(self._nnz):
            col_counts[self._indices[k]] += 1

        # Build indptr
        csc_indptr = IndexArray(self._cols + 1)
        csc_indptr[0] = 0
        for j in range(self._cols):
            csc_indptr[j + 1] = csc_indptr[j] + col_counts[j]

        # Fill data
        csc_data = RealArray(self._nnz)
        csc_indices = IndexArray(self._nnz)
        col_pos = [csc_indptr[j] for j in range(self._cols)]

        for i in range(self._rows):
            start = self._indptr[i]
            end = self._indptr[i + 1]
            for k in range(start, end):
                j = self._indices[k]
                pos = col_pos[j]
                csc_data[pos] = self._data[k]
                csc_indices[pos] = i
                col_pos[j] += 1

        return SclCSC.from_arrays(csc_data, csc_indices, csc_indptr, self.shape)

    @property
    def T(self) -> "SclCSC":
        """Transpose (returns CSC)."""
        return self.to_csc()

    # -------------------------------------------------------------------------
    # Reordering
    # -------------------------------------------------------------------------

    def reorder_rows(self, order: Sequence[int]) -> "SclCSR":
        """
        Reorder rows.

        Args:
            order: New row order (list of old indices)

        Returns:
            Reordered matrix
        """
        self.materialize()

        new_rows = len(order)
        new_data = []
        new_indices = []
        new_indptr = [0]

        for old_i in order:
            start = self._indptr[old_i]
            end = self._indptr[old_i + 1]
            for k in range(start, end):
                new_data.append(self._data[k])
                new_indices.append(self._indices[k])
            new_indptr.append(len(new_data))

        return SclCSR.from_arrays(new_data, new_indices, new_indptr, (new_rows, self._cols))

    def reorder_cols(self, order: Sequence[int]) -> "SclCSR":
        """
        Reorder columns.

        Args:
            order: New column order (list of old indices)

        Returns:
            Reordered matrix
        """
        self.materialize()

        new_cols = len(order)
        col_remap = {old: new for new, old in enumerate(order)}

        new_data = []
        new_indices = []
        new_indptr = [0]

        for i in range(self._rows):
            start = self._indptr[i]
            end = self._indptr[i + 1]

            # Collect and remap columns for this row
            row_entries = []
            for k in range(start, end):
                old_col = self._indices[k]
                if old_col in col_remap:
                    row_entries.append((col_remap[old_col], self._data[k]))

            # Sort by new column index
            row_entries.sort(key=lambda x: x[0])

            for new_col, val in row_entries:
                new_indices.append(new_col)
                new_data.append(val)

            new_indptr.append(len(new_data))

        return SclCSR.from_arrays(new_data, new_indices, new_indptr, (self._rows, new_cols))


# =============================================================================
# SclCSC - CSC Sparse Matrix
# =============================================================================

class SclCSC(SparseBase):
    """
    Compressed Sparse Column (CSC) Matrix.

    Column-major storage: efficient for column operations.
    Symmetric implementation with SclCSR.
    """

    _format = "csc"
    _primary_axis = 1

    def _primary_size(self) -> int:
        return self._cols

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_arrays(cls, data: Union[RealArray, Sequence[float]],
                    indices: Union[IndexArray, Sequence[int]],
                    indptr: Union[IndexArray, Sequence[int]],
                    shape: Tuple[int, int],
                    copy: bool = False) -> "SclCSC":
        """
        Create from raw arrays.

        Args:
            data: Non-zero values
            indices: Row indices
            indptr: Column pointers
            shape: (rows, cols)
            copy: Whether to copy input arrays
        """
        obj = cls()
        obj._rows, obj._cols = shape

        if isinstance(data, RealArray):
            obj._data = data.copy() if copy else data
        else:
            obj._data = RealArray.from_sequence(data)

        if isinstance(indices, IndexArray):
            obj._indices = indices.copy() if copy else indices
        else:
            obj._indices = IndexArray.from_sequence(indices)

        if isinstance(indptr, IndexArray):
            obj._indptr = indptr.copy() if copy else indptr
        else:
            obj._indptr = IndexArray.from_sequence(indptr)

        obj._nnz = obj._data.size
        obj._storage = CustomStorage(
            ownership=Ownership.OWNED,
            memory_bytes=obj._data.nbytes + obj._indices.nbytes + obj._indptr.nbytes,
        )

        return obj

    @classmethod
    def from_scipy(cls, sp_matrix) -> "SclCSC":
        """Create from scipy.sparse matrix."""
        from scipy import sparse

        if not sparse.isspmatrix_csc(sp_matrix):
            sp_matrix = sp_matrix.tocsc()

        from scl.array import from_numpy
        import numpy as np

        data = from_numpy(np.ascontiguousarray(sp_matrix.data, dtype=np.float64))
        indices = from_numpy(np.ascontiguousarray(sp_matrix.indices, dtype=np.int64))
        indptr = from_numpy(np.ascontiguousarray(sp_matrix.indptr, dtype=np.int64))

        return cls.from_arrays(data, indices, indptr, sp_matrix.shape)

    @classmethod
    def from_adapter(cls, adapter: SparseAdapter) -> "SclCSC":
        """Create from Python adapter."""
        obj = cls()
        obj._adapter = adapter
        obj._rows, obj._cols = adapter.shape()
        obj._nnz = adapter.nnz()
        obj._storage = VirtualStorage(source_type=SourceType.PARENT)

        return obj

    @classmethod
    def empty(cls, shape: Tuple[int, int]) -> "SclCSC":
        """Create empty matrix with given shape."""
        rows, cols = shape
        data = RealArray(0)
        indices = IndexArray(0)
        indptr = IndexArray(cols + 1)
        indptr.zero()
        return cls.from_arrays(data, indices, indptr, shape)

    # -------------------------------------------------------------------------
    # Element Access
    # -------------------------------------------------------------------------

    def _get_element(self, i: int, j: int) -> float:
        """Get single element value."""
        self.materialize()

        start = self._indptr[j]
        end = self._indptr[j + 1]
        for k in range(start, end):
            if self._indices[k] == i:
                return self._data[k]
        return 0.0

    def __getitem__(self, key):
        """Slice operation."""
        if isinstance(key, tuple):
            row_key, col_key = key if len(key) == 2 else (key[0], None)
        else:
            row_key = key
            col_key = None

        if isinstance(row_key, int) and isinstance(col_key, int):
            return self._get_element(row_key, col_key)

        if not self.lazy_slicing:
            return self._slice_eager(row_key, col_key)

        return self._slice_lazy(row_key, col_key)

    def _slice_lazy(self, row_key, col_key) -> "SclCSC":
        """Create lazy view."""
        view = SclCSC()
        view._storage = VirtualStorage(source_type=SourceType.SLICE)
        view._storage.add_chunk(self, (0, self._rows), (0, self._cols))
        view._config_overrides = self._config_overrides.copy()

        if row_key is not None and not (isinstance(row_key, slice) and row_key == slice(None)):
            view._storage.row_mask = self._to_mask(row_key, self._rows)

        if col_key is not None and not (isinstance(col_key, slice) and col_key == slice(None)):
            view._storage.col_mask = self._to_mask(col_key, self._cols)

        row_mask = view._storage.row_mask
        col_mask = view._storage.col_mask
        view._rows = row_mask.count_nonzero() if row_mask else self._rows
        view._cols = col_mask.count_nonzero() if col_mask else self._cols
        view._nnz = self._nnz

        return view

    def _slice_eager(self, row_key, col_key) -> "SclCSC":
        """Create eager copy of slice."""
        self.materialize()

        row_mask = None if row_key is None or (isinstance(row_key, slice) and row_key == slice(None)) else self._to_mask(row_key, self._rows)
        col_mask = None if col_key is None or (isinstance(col_key, slice) and col_key == slice(None)) else self._to_mask(col_key, self._cols)

        return self._extract_submatrix(row_mask, col_mask)

    def _to_mask(self, key, length: int) -> ByteArray:
        """Convert key to uint8 mask."""
        mask = ByteArray(length)

        if isinstance(key, slice):
            for i in range(*key.indices(length)):
                mask[i] = 1
        elif isinstance(key, (list, tuple)):
            for i in key:
                if isinstance(i, bool):
                    if i:
                        idx = list(key).index(i)
                        mask[idx] = 1
                else:
                    mask[i] = 1
        elif hasattr(key, '__iter__'):
            for i, v in enumerate(key):
                if isinstance(v, bool):
                    if v:
                        mask[i] = 1
                else:
                    mask[v] = 1
        else:
            mask[int(key)] = 1

        return mask

    def _extract_submatrix(self, row_mask: Optional[ByteArray],
                           col_mask: Optional[ByteArray]) -> "SclCSC":
        """Extract submatrix based on masks."""
        col_indices = [j for j in range(self._cols) if col_mask is None or col_mask[j]]
        row_set = None
        row_remap = None

        if row_mask is not None:
            row_set = set(i for i in range(self._rows) if row_mask[i])
            row_remap = {old: new for new, old in enumerate(sorted(row_set))}

        new_data = []
        new_indices = []
        new_indptr = [0]

        for j in col_indices:
            start = self._indptr[j]
            end = self._indptr[j + 1]
            for k in range(start, end):
                row = self._indices[k]
                if row_set is None or row in row_set:
                    new_data.append(self._data[k])
                    if row_remap:
                        new_indices.append(row_remap[row])
                    else:
                        new_indices.append(row)
            new_indptr.append(len(new_data))

        new_rows = len(row_set) if row_set else self._rows
        new_cols = len(col_indices)

        return SclCSC.from_arrays(new_data, new_indices, new_indptr, (new_rows, new_cols))

    # -------------------------------------------------------------------------
    # Column Access
    # -------------------------------------------------------------------------

    def getcol(self, j: int) -> Tuple[RealArray, IndexArray]:
        """
        Get column data efficiently.

        Args:
            j: Column index

        Returns:
            (values, row_indices)
        """
        self.materialize()

        if j < 0:
            j += self._cols
        if j < 0 or j >= self._cols:
            raise IndexError(f"Column index {j} out of range [0, {self._cols})")

        start = self._indptr[j]
        end = self._indptr[j + 1]
        length = end - start

        if length == 0:
            return RealArray(0), IndexArray(0)

        vals = RealArray(length)
        rows = IndexArray(length)
        for k in range(length):
            vals[k] = self._data[start + k]
            rows[k] = self._indices[start + k]

        return vals, rows

    def col_length(self, j: int) -> int:
        """Get number of non-zeros in column."""
        self.materialize()
        return self._indptr[j + 1] - self._indptr[j]

    # -------------------------------------------------------------------------
    # Materialization
    # -------------------------------------------------------------------------

    def _materialize_from_adapter(self):
        """Materialize from Python adapter."""
        storage = self._storage
        if isinstance(storage, VirtualStorage):
            row_mask = storage.row_mask
            col_mask = storage.col_mask
        else:
            row_mask = None
            col_mask = None

        if col_mask is not None:
            col_indices = [j for j in range(self._cols) if col_mask[j]]
        else:
            col_indices = list(range(self._cols))

        data, indices, indptr = self._adapter.load_cols(col_indices)

        if row_mask is not None:
            row_set = set(i for i in range(self._rows) if row_mask[i])
            row_remap = {old: new for new, old in enumerate(sorted(row_set))}

            new_data = []
            new_indices = []
            new_indptr = [0]

            for j in range(len(col_indices)):
                start = indptr[j]
                end = indptr[j + 1]
                for k in range(start, end):
                    row = indices[k]
                    if row in row_set:
                        new_data.append(data[k])
                        new_indices.append(row_remap[row])
                new_indptr.append(len(new_data))

            self._data = RealArray.from_sequence(new_data)
            self._indices = IndexArray.from_sequence(new_indices)
            self._indptr = IndexArray.from_sequence(new_indptr)
        else:
            self._data = data if isinstance(data, RealArray) else RealArray.from_sequence(data)
            self._indices = indices if isinstance(indices, IndexArray) else IndexArray.from_sequence(indices)
            self._indptr = indptr if isinstance(indptr, IndexArray) else IndexArray.from_sequence(indptr)

        self._cols = len(col_indices)
        self._rows = row_mask.count_nonzero() if row_mask else self._rows
        self._nnz = self._data.size

        self._storage = CustomStorage(ownership=Ownership.OWNED)
        self._adapter = None

    def _materialize_from_mapped(self):
        """Materialize from memory-mapped file (not implemented for CSC)."""
        raise NotImplementedError("CSC from mapped file not yet implemented")

    def _materialize_from_virtual(self):
        """Materialize from virtual storage."""
        storage = self._storage
        if not isinstance(storage, VirtualStorage):
            return

        if not storage.chunks:
            return

        chunk = storage.chunks[0]
        source = chunk.get_source()
        if source is None:
            raise RuntimeError("Source matrix has been garbage collected")

        source.materialize()

        row_mask = storage.row_mask
        col_mask = storage.col_mask

        if row_mask is None and col_mask is None:
            self._data = source._data.copy()
            self._indices = source._indices.copy()
            self._indptr = source._indptr.copy()
            self._nnz = source._nnz
        else:
            result = source._extract_submatrix(row_mask, col_mask)
            self._data = result._data
            self._indices = result._indices
            self._indptr = result._indptr
            self._rows = result._rows
            self._cols = result._cols
            self._nnz = result._nnz

        self._storage = CustomStorage(ownership=Ownership.OWNED)

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def iter_cols(self) -> Iterator[Tuple[int, RealArray, IndexArray]]:
        """Iterate over columns."""
        self.materialize()
        for j in range(self._cols):
            vals, rows = self.getcol(j)
            yield j, vals, rows

    def iter_primary(self) -> Iterator[Tuple[RealArray, IndexArray]]:
        """Iterate over primary axis (columns)."""
        self.materialize()
        for j in range(self._cols):
            yield self.getcol(j)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def sum(self, axis: Optional[int] = None) -> Union[float, RealArray]:
        """Sum of elements."""
        self.materialize()

        if axis is None:
            total = 0.0
            for v in self._data:
                total += v
            return total

        elif axis == 0:  # Column sums
            result = RealArray(self._cols)
            for j in range(self._cols):
                start = self._indptr[j]
                end = self._indptr[j + 1]
                col_sum = 0.0
                for k in range(start, end):
                    col_sum += self._data[k]
                result[j] = col_sum
            return result

        elif axis == 1:  # Row sums
            result = RealArray(self._rows)
            result.zero()
            for j in range(self._cols):
                start = self._indptr[j]
                end = self._indptr[j + 1]
                for k in range(start, end):
                    result[self._indices[k]] += self._data[k]
            return result

        raise ValueError(f"Invalid axis: {axis}")

    def mean(self, axis: Optional[int] = None) -> Union[float, RealArray]:
        """Mean of elements."""
        sums = self.sum(axis)
        if axis is None:
            return sums / (self._rows * self._cols)
        elif axis == 0:
            for i in range(sums.size):
                sums[i] /= self._rows
            return sums
        else:
            for i in range(sums.size):
                sums[i] /= self._cols
            return sums

    def min(self, axis: Optional[int] = None) -> Union[float, RealArray]:
        """Minimum value."""
        self.materialize()

        if axis is None:
            min_val = float('inf')
            for v in self._data:
                if v < min_val:
                    min_val = v
            if self._nnz < self._rows * self._cols and min_val > 0:
                min_val = 0.0
            return min_val

        elif axis == 0:  # Column min
            result = RealArray(self._cols)
            for j in range(self._cols):
                start = self._indptr[j]
                end = self._indptr[j + 1]
                if start == end:
                    result[j] = 0.0
                else:
                    col_min = float('inf')
                    for k in range(start, end):
                        if self._data[k] < col_min:
                            col_min = self._data[k]
                    if (end - start) < self._rows and col_min > 0:
                        col_min = 0.0
                    result[j] = col_min
            return result

        elif axis == 1:  # Row min
            result = RealArray(self._rows)
            counts = [0] * self._rows
            for i in range(self._rows):
                result[i] = float('inf')

            for j in range(self._cols):
                start = self._indptr[j]
                end = self._indptr[j + 1]
                for k in range(start, end):
                    i = self._indices[k]
                    if self._data[k] < result[i]:
                        result[i] = self._data[k]
                    counts[i] += 1

            for i in range(self._rows):
                if counts[i] < self._cols and result[i] > 0:
                    result[i] = 0.0
                elif counts[i] == 0:
                    result[i] = 0.0

            return result

        raise ValueError(f"Invalid axis: {axis}")

    def max(self, axis: Optional[int] = None) -> Union[float, RealArray]:
        """Maximum value."""
        self.materialize()

        if axis is None:
            max_val = float('-inf')
            for v in self._data:
                if v > max_val:
                    max_val = v
            if self._nnz < self._rows * self._cols and max_val < 0:
                max_val = 0.0
            return max_val

        elif axis == 0:  # Column max
            result = RealArray(self._cols)
            for j in range(self._cols):
                start = self._indptr[j]
                end = self._indptr[j + 1]
                if start == end:
                    result[j] = 0.0
                else:
                    col_max = float('-inf')
                    for k in range(start, end):
                        if self._data[k] > col_max:
                            col_max = self._data[k]
                    if (end - start) < self._rows and col_max < 0:
                        col_max = 0.0
                    result[j] = col_max
            return result

        elif axis == 1:  # Row max
            result = RealArray(self._rows)
            counts = [0] * self._rows
            for i in range(self._rows):
                result[i] = float('-inf')

            for j in range(self._cols):
                start = self._indptr[j]
                end = self._indptr[j + 1]
                for k in range(start, end):
                    i = self._indices[k]
                    if self._data[k] > result[i]:
                        result[i] = self._data[k]
                    counts[i] += 1

            for i in range(self._rows):
                if counts[i] < self._cols and result[i] < 0:
                    result[i] = 0.0
                elif counts[i] == 0:
                    result[i] = 0.0

            return result

        raise ValueError(f"Invalid axis: {axis}")

    def nonzero(self) -> Tuple[IndexArray, IndexArray]:
        """Return indices of non-zero elements."""
        self.materialize()

        rows = IndexArray(self._nnz)
        cols = IndexArray(self._nnz)

        idx = 0
        for j in range(self._cols):
            start = self._indptr[j]
            end = self._indptr[j + 1]
            for k in range(start, end):
                rows[idx] = self._indices[k]
                cols[idx] = j
                idx += 1

        return rows, cols

    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------

    def __neg__(self) -> "SclCSC":
        """Negation: -A"""
        self.materialize()
        new_data = RealArray(self._nnz)
        for k in range(self._nnz):
            new_data[k] = -self._data[k]
        return SclCSC.from_arrays(new_data, self._indices.copy(), self._indptr.copy(), self.shape)

    def __mul__(self, other) -> "SclCSC":
        """Element-wise multiplication with scalar."""
        if isinstance(other, (int, float)):
            self.materialize()
            new_data = RealArray(self._nnz)
            for k in range(self._nnz):
                new_data[k] = self._data[k] * other
            return SclCSC.from_arrays(new_data, self._indices.copy(), self._indptr.copy(), self.shape)
        raise NotImplementedError("Element-wise matrix multiplication not implemented")

    def __rmul__(self, other) -> "SclCSC":
        """Right multiplication with scalar."""
        return self.__mul__(other)

    def __truediv__(self, other) -> "SclCSC":
        """Element-wise division by scalar."""
        if isinstance(other, (int, float)):
            return self.__mul__(1.0 / other)
        raise NotImplementedError("Element-wise matrix division not implemented")

    # -------------------------------------------------------------------------
    # Conversion
    # -------------------------------------------------------------------------

    def copy(self) -> "SclCSC":
        """Create a deep copy."""
        self.materialize()
        return SclCSC.from_arrays(
            self._data.copy(),
            self._indices.copy(),
            self._indptr.copy(),
            self.shape
        )

    def to_scipy(self):
        """Convert to scipy.sparse.csc_matrix."""
        self.materialize()
        from scipy import sparse
        from scl.array import to_numpy

        return sparse.csc_matrix(
            (to_numpy(self._data), to_numpy(self._indices), to_numpy(self._indptr)),
            shape=self.shape
        )

    def to_dense(self) -> RealArray:
        """Convert to dense (row-major) array."""
        self.materialize()
        dense = RealArray(self._rows * self._cols)
        dense.zero()

        for j in range(self._cols):
            start = self._indptr[j]
            end = self._indptr[j + 1]
            for k in range(start, end):
                dense[self._indices[k] * self._cols + j] = self._data[k]

        return dense

    def to_csr(self) -> SclCSR:
        """Convert to CSR format."""
        self.materialize()

        # Count elements per row
        row_counts = [0] * self._rows
        for k in range(self._nnz):
            row_counts[self._indices[k]] += 1

        # Build indptr
        csr_indptr = IndexArray(self._rows + 1)
        csr_indptr[0] = 0
        for i in range(self._rows):
            csr_indptr[i + 1] = csr_indptr[i] + row_counts[i]

        # Fill data
        csr_data = RealArray(self._nnz)
        csr_indices = IndexArray(self._nnz)
        row_pos = [csr_indptr[i] for i in range(self._rows)]

        for j in range(self._cols):
            start = self._indptr[j]
            end = self._indptr[j + 1]
            for k in range(start, end):
                i = self._indices[k]
                pos = row_pos[i]
                csr_data[pos] = self._data[k]
                csr_indices[pos] = j
                row_pos[i] += 1

        return SclCSR.from_arrays(csr_data, csr_indices, csr_indptr, self.shape)

    @property
    def T(self) -> SclCSR:
        """Transpose (returns CSR)."""
        return self.to_csr()

    # -------------------------------------------------------------------------
    # Reordering
    # -------------------------------------------------------------------------

    def reorder_rows(self, order: Sequence[int]) -> "SclCSC":
        """Reorder rows."""
        self.materialize()

        new_rows = len(order)
        row_remap = {old: new for new, old in enumerate(order)}

        new_data = []
        new_indices = []
        new_indptr = [0]

        for j in range(self._cols):
            start = self._indptr[j]
            end = self._indptr[j + 1]

            col_entries = []
            for k in range(start, end):
                old_row = self._indices[k]
                if old_row in row_remap:
                    col_entries.append((row_remap[old_row], self._data[k]))

            col_entries.sort(key=lambda x: x[0])

            for new_row, val in col_entries:
                new_indices.append(new_row)
                new_data.append(val)

            new_indptr.append(len(new_data))

        return SclCSC.from_arrays(new_data, new_indices, new_indptr, (new_rows, self._cols))

    def reorder_cols(self, order: Sequence[int]) -> "SclCSC":
        """Reorder columns."""
        self.materialize()

        new_cols = len(order)
        new_data = []
        new_indices = []
        new_indptr = [0]

        for old_j in order:
            start = self._indptr[old_j]
            end = self._indptr[old_j + 1]
            for k in range(start, end):
                new_data.append(self._data[k])
                new_indices.append(self._indices[k])
            new_indptr.append(len(new_data))

        return SclCSC.from_arrays(new_data, new_indices, new_indptr, (self._rows, new_cols))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Level control
    "LazyLevel",
    # Adapter protocol
    "SparseAdapter",
    # Base class
    "SparseBase",
    # Concrete classes
    "SclCSR",
    "SclCSC",
]
