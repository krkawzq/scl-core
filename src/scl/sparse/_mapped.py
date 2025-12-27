"""
Memory-Mapped Sparse Matrices

MappedCustomCSR/CSC are sparse matrices backed by memory-mapped files.
They require three binary files: data.bin, indices.bin, indptr.bin.

C++ Equivalent:
    scl::io::MappedCustomSparse<T, IsCSR> from scl/io/mmatrix.hpp

Memory Model:
    - Files are memory-mapped (not fully loaded into RAM)
    - On-demand page loading by OS
    - Suitable for datasets larger than available RAM

Use Cases:
    - Large datasets that don't fit in memory
    - Out-of-core processing
    - Lazy loading from disk

Initialization:
    Requires paths to three binary files:
    - data_path: Non-zero values (nnz * sizeof(Real))
    - indices_path: Secondary indices (nnz * sizeof(Index))
    - indptr_path: Primary pointers ((primary_dim + 1) * sizeof(Index))

Example:
    >>> from scl.sparse import MappedCustomCSR
    >>> 
    >>> # Create from three binary files
    >>> mat = MappedCustomCSR(
    ...     data_path="data.bin",
    ...     indices_path="indices.bin", 
    ...     indptr_path="indptr.bin",
    ...     shape=(1000000, 50000),
    ...     dtype='float64'
    ... )
    >>> 
    >>> # Access rows (lazy loading)
    >>> row_data = mat.row_values(0)
    >>> 
    >>> # Materialize subset
    >>> subset = mat[0:1000, :].materialize()
"""

from typing import Tuple, Optional, Union, TYPE_CHECKING
from pathlib import Path
import numpy as np
import mmap
import os

from ._base import CSRBase, CSCBase
from ._array import Array, zeros

if TYPE_CHECKING:
    from ._custom import CustomCSR, CustomCSC
    from ._virtual import VirtualCSR, VirtualCSC
    from scipy.sparse import spmatrix

__all__ = ['MappedCustomCSR', 'MappedCustomCSC', 'MappedVirtualCSR', 'MappedVirtualCSC']


# =============================================================================
# Internal Marker for MappedVirtual
# =============================================================================

class _MappedVirtualInternal:
    """Internal marker to prevent external construction."""
    pass


_MAPPED_VIRTUAL_KEY = _MappedVirtualInternal()


class MappedCustomCSR(CSRBase):
    """
    Memory-mapped CSR sparse matrix.
    
    Data is stored in three binary files and memory-mapped for
    efficient out-of-core access.
    
    C++ Equivalent:
        scl::io::MappedCustomSparse<Real, true>
    
    File Format:
        - data.bin: Raw binary array of float32/float64 values
        - indices.bin: Raw binary array of int64 column indices
        - indptr.bin: Raw binary array of int64 row pointers
    
    Memory Model:
        Files are memory-mapped, not fully loaded. The OS handles
        page loading on demand. This allows working with datasets
        larger than available RAM.
    
    Example:
        >>> mat = MappedCustomCSR(
        ...     data_path="matrix/data.bin",
        ...     indices_path="matrix/indices.bin",
        ...     indptr_path="matrix/indptr.bin", 
        ...     shape=(1000000, 50000),
        ...     dtype='float64'
        ... )
        >>> 
        >>> # Lazy row access
        >>> vals = mat.row_values(0)
        >>> 
        >>> # Convert to CustomCSR (loads all data)
        >>> custom = mat.materialize()
    """
    
    __slots__ = (
        '_data_path', '_indices_path', '_indptr_path',
        '_data_mmap', '_indices_mmap', '_indptr_mmap',
        '_data_file', '_indices_file', '_indptr_file',
        '_data_np', '_indices_np', '_indptr_np',
        '_shape', '_dtype', '_nnz'
    )
    
    def __init__(
        self,
        data_path: Union[str, Path],
        indices_path: Union[str, Path],
        indptr_path: Union[str, Path],
        shape: Tuple[int, int],
        dtype: str = 'float64',
        nnz: Optional[int] = None
    ):
        """Initialize MappedCustomCSR from binary files.
        
        Args:
            data_path: Path to data binary file (values)
            indices_path: Path to indices binary file (column indices)
            indptr_path: Path to indptr binary file (row pointers)
            shape: Matrix dimensions (rows, cols)
            dtype: Data type ('float32' or 'float64')
            nnz: Number of non-zeros (inferred from file size if None)
        """
        self._data_path = Path(data_path)
        self._indices_path = Path(indices_path)
        self._indptr_path = Path(indptr_path)
        self._shape = tuple(shape)
        self._dtype = dtype
        
        # Validate files exist
        for path in [self._data_path, self._indices_path, self._indptr_path]:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
        
        # Determine dtype sizes
        np_dtype = np.float32 if dtype == 'float32' else np.float64
        value_size = np_dtype().itemsize
        index_size = np.int64().itemsize
        
        # Memory map the files
        self._data_file = open(self._data_path, 'rb')
        self._indices_file = open(self._indices_path, 'rb')
        self._indptr_file = open(self._indptr_path, 'rb')
        
        self._data_mmap = mmap.mmap(
            self._data_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        self._indices_mmap = mmap.mmap(
            self._indices_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        self._indptr_mmap = mmap.mmap(
            self._indptr_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        
        # Create numpy views over mapped memory
        data_size = os.path.getsize(self._data_path)
        indices_size = os.path.getsize(self._indices_path)
        indptr_size = os.path.getsize(self._indptr_path)
        
        inferred_nnz = data_size // value_size
        expected_indptr_count = shape[0] + 1
        
        if indptr_size // index_size != expected_indptr_count:
            raise ValueError(
                f"indptr file size mismatch: expected {expected_indptr_count * index_size} bytes, "
                f"got {indptr_size} bytes"
            )
        
        self._data_np = np.frombuffer(self._data_mmap, dtype=np_dtype)
        self._indices_np = np.frombuffer(self._indices_mmap, dtype=np.int64)
        self._indptr_np = np.frombuffer(self._indptr_mmap, dtype=np.int64)
        
        # Verify nnz
        if nnz is not None:
            if nnz != inferred_nnz:
                raise ValueError(f"nnz mismatch: provided {nnz}, inferred {inferred_nnz}")
            self._nnz = nnz
        else:
            self._nnz = inferred_nnz
        
        # Verify with indptr
        if len(self._indptr_np) > 0:
            indptr_nnz = int(self._indptr_np[-1])
            if indptr_nnz != self._nnz:
                raise ValueError(
                    f"nnz mismatch: indptr indicates {indptr_nnz}, file has {self._nnz}"
                )
    
    def __del__(self):
        """Cleanup memory mappings."""
        self.close()
    
    def close(self):
        """Close memory mappings and file handles."""
        for mmap_obj in [self._data_mmap, self._indices_mmap, self._indptr_mmap]:
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except Exception:
                    pass
        
        for file_obj in [self._data_file, self._indices_file, self._indptr_file]:
            if file_obj is not None:
                try:
                    file_obj.close()
                except Exception:
                    pass
        
        self._data_mmap = None
        self._indices_mmap = None
        self._indptr_mmap = None
        self._data_file = None
        self._indices_file = None
        self._indptr_file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
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
    
    @property
    def data(self) -> np.ndarray:
        """Memory-mapped data array (read-only numpy view)."""
        return self._data_np
    
    @property
    def indices(self) -> np.ndarray:
        """Memory-mapped indices array (read-only numpy view)."""
        return self._indices_np
    
    @property
    def indptr(self) -> np.ndarray:
        """Memory-mapped indptr array (read-only numpy view)."""
        return self._indptr_np
    
    # =========================================================================
    # CSRBase Interface
    # =========================================================================
    
    def row_values(self, i: int) -> Array:
        """Get non-zero values for row i."""
        start = int(self._indptr_np[i])
        end = int(self._indptr_np[i + 1])
        
        if end == start:
            return zeros(0, dtype=self._dtype)
        
        # Return view into mapped data
        return Array.from_numpy(self._data_np[start:end], copy=False)
    
    def row_indices(self, i: int) -> Array:
        """Get column indices for row i."""
        start = int(self._indptr_np[i])
        end = int(self._indptr_np[i + 1])
        
        if end == start:
            return zeros(0, dtype='int64')
        
        return Array.from_numpy(self._indices_np[start:end], copy=False)
    
    def row_length(self, i: int) -> int:
        """Get number of non-zeros in row i."""
        return int(self._indptr_np[i + 1] - self._indptr_np[i])
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def sum(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute sum along axis."""
        if axis is None:
            return float(np.sum(self._data_np))
        elif axis == 1:
            result = np.zeros(self.rows, dtype=np.float64)
            for i in range(self.rows):
                start = int(self._indptr_np[i])
                end = int(self._indptr_np[i + 1])
                result[i] = np.sum(self._data_np[start:end])
            return result
        else:  # axis == 0
            result = np.zeros(self.cols, dtype=np.float64)
            for k in range(self._nnz):
                result[self._indices_np[k]] += self._data_np[k]
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
        """Convert to CustomCSR (loads all data into memory).
        
        Returns:
            CustomCSR with copied data
        """
        from ._custom import CustomCSR
        
        return CustomCSR(
            data=Array.from_numpy(self._data_np.copy(), copy=False),
            indices=Array.from_numpy(self._indices_np.copy(), copy=False),
            indptr=Array.from_numpy(self._indptr_np.copy(), copy=False),
            shape=self._shape,
            dtype=self._dtype
        )
    
    def to_scipy(self) -> 'spmatrix':
        """Convert to scipy CSR matrix (loads all data)."""
        return self.materialize().to_scipy()
    
    def copy(self) -> 'CustomCSR':
        """Create materialized copy."""
        return self.materialize()
    
    # =========================================================================
    # Slicing (produces MappedVirtualCSR)
    # =========================================================================
    
    def slice_rows(self, row_indices: np.ndarray) -> 'MappedVirtualCSR':
        """Create zero-copy view with row selection.
        
        Args:
            row_indices: Array of row indices to select
            
        Returns:
            MappedVirtualCSR view
        """
        return MappedVirtualCSR._from_mapped_custom_slice(self, row_indices)
    


class MappedCustomCSC(CSCBase):
    """
    Memory-mapped CSC sparse matrix.
    
    Column-oriented equivalent of MappedCustomCSR.
    
    C++ Equivalent:
        scl::io::MappedCustomSparse<Real, false>
    """
    
    __slots__ = (
        '_data_path', '_indices_path', '_indptr_path',
        '_data_mmap', '_indices_mmap', '_indptr_mmap',
        '_data_file', '_indices_file', '_indptr_file',
        '_data_np', '_indices_np', '_indptr_np',
        '_shape', '_dtype', '_nnz'
    )
    
    def __init__(
        self,
        data_path: Union[str, Path],
        indices_path: Union[str, Path],
        indptr_path: Union[str, Path],
        shape: Tuple[int, int],
        dtype: str = 'float64',
        nnz: Optional[int] = None
    ):
        """Initialize MappedCustomCSC from binary files."""
        self._data_path = Path(data_path)
        self._indices_path = Path(indices_path)
        self._indptr_path = Path(indptr_path)
        self._shape = tuple(shape)
        self._dtype = dtype
        
        for path in [self._data_path, self._indices_path, self._indptr_path]:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
        
        np_dtype = np.float32 if dtype == 'float32' else np.float64
        value_size = np_dtype().itemsize
        index_size = np.int64().itemsize
        
        self._data_file = open(self._data_path, 'rb')
        self._indices_file = open(self._indices_path, 'rb')
        self._indptr_file = open(self._indptr_path, 'rb')
        
        self._data_mmap = mmap.mmap(
            self._data_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        self._indices_mmap = mmap.mmap(
            self._indices_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        self._indptr_mmap = mmap.mmap(
            self._indptr_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        
        data_size = os.path.getsize(self._data_path)
        indptr_size = os.path.getsize(self._indptr_path)
        
        inferred_nnz = data_size // value_size
        expected_indptr_count = shape[1] + 1  # cols + 1 for CSC
        
        if indptr_size // index_size != expected_indptr_count:
            raise ValueError(
                f"indptr file size mismatch: expected {expected_indptr_count * index_size} bytes, "
                f"got {indptr_size} bytes"
            )
        
        self._data_np = np.frombuffer(self._data_mmap, dtype=np_dtype)
        self._indices_np = np.frombuffer(self._indices_mmap, dtype=np.int64)
        self._indptr_np = np.frombuffer(self._indptr_mmap, dtype=np.int64)
        
        if nnz is not None:
            if nnz != inferred_nnz:
                raise ValueError(f"nnz mismatch: provided {nnz}, inferred {inferred_nnz}")
            self._nnz = nnz
        else:
            self._nnz = inferred_nnz
        
        if len(self._indptr_np) > 0:
            indptr_nnz = int(self._indptr_np[-1])
            if indptr_nnz != self._nnz:
                raise ValueError(
                    f"nnz mismatch: indptr indicates {indptr_nnz}, file has {self._nnz}"
                )
    
    def __del__(self):
        self.close()
    
    def close(self):
        """Close memory mappings and file handles."""
        for mmap_obj in [self._data_mmap, self._indices_mmap, self._indptr_mmap]:
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except Exception:
                    pass
        
        for file_obj in [self._data_file, self._indices_file, self._indptr_file]:
            if file_obj is not None:
                try:
                    file_obj.close()
                except Exception:
                    pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
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
    
    @property
    def data(self) -> np.ndarray:
        return self._data_np
    
    @property
    def indices(self) -> np.ndarray:
        return self._indices_np
    
    @property
    def indptr(self) -> np.ndarray:
        return self._indptr_np
    
    # =========================================================================
    # CSCBase Interface
    # =========================================================================
    
    def col_values(self, j: int) -> Array:
        """Get non-zero values for column j."""
        start = int(self._indptr_np[j])
        end = int(self._indptr_np[j + 1])
        
        if end == start:
            return zeros(0, dtype=self._dtype)
        
        return Array.from_numpy(self._data_np[start:end], copy=False)
    
    def col_indices(self, j: int) -> Array:
        """Get row indices for column j."""
        start = int(self._indptr_np[j])
        end = int(self._indptr_np[j + 1])
        
        if end == start:
            return zeros(0, dtype='int64')
        
        return Array.from_numpy(self._indices_np[start:end], copy=False)
    
    def col_length(self, j: int) -> int:
        """Get number of non-zeros in column j."""
        return int(self._indptr_np[j + 1] - self._indptr_np[j])
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def sum(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute sum along axis."""
        if axis is None:
            return float(np.sum(self._data_np))
        elif axis == 0:
            result = np.zeros(self.cols, dtype=np.float64)
            for j in range(self.cols):
                start = int(self._indptr_np[j])
                end = int(self._indptr_np[j + 1])
                result[j] = np.sum(self._data_np[start:end])
            return result
        else:  # axis == 1
            result = np.zeros(self.rows, dtype=np.float64)
            for k in range(self._nnz):
                result[self._indices_np[k]] += self._data_np[k]
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
        """Convert to CustomCSC (loads all data into memory)."""
        from ._custom import CustomCSC
        
        return CustomCSC(
            data=Array.from_numpy(self._data_np.copy(), copy=False),
            indices=Array.from_numpy(self._indices_np.copy(), copy=False),
            indptr=Array.from_numpy(self._indptr_np.copy(), copy=False),
            shape=self._shape,
            dtype=self._dtype
        )
    
    def to_scipy(self) -> 'spmatrix':
        """Convert to scipy CSC matrix (loads all data)."""
        return self.materialize().to_scipy()
    
    def copy(self) -> 'CustomCSC':
        """Create materialized copy."""
        return self.materialize()
    
    def slice_cols(self, col_indices: np.ndarray) -> 'MappedVirtualCSC':
        """Create zero-copy view with column selection."""
        return MappedVirtualCSC._from_mapped_custom_slice(self, col_indices)


# =============================================================================
# MappedVirtualCSR: Zero-Copy Row Slicing over Mapped Data
# =============================================================================

class MappedVirtualCSR(CSRBase):
    """
    Memory-mapped virtual CSR matrix - zero-copy view with index remapping.
    
    INTERNAL USE ONLY: Cannot be constructed externally.
    Created by MappedCustomCSR.slice_rows().
    
    C++ Equivalent:
        scl::io::MappedVirtualSparse<Real, true>
    """
    
    __slots__ = ('_source', '_map', '_shape', '_dtype', '_nnz')
    
    def __init__(
        self,
        source: 'MappedCustomCSR',
        index_map: np.ndarray,
        shape: Tuple[int, int],
        nnz: int,
        _internal_key=None
    ):
        """Initialize MappedVirtualCSR (INTERNAL ONLY)."""
        if _internal_key is not _MAPPED_VIRTUAL_KEY:
            raise TypeError(
                "MappedVirtualCSR cannot be constructed directly. "
                "Use MappedCustomCSR.slice_rows() instead."
            )
        
        self._source = source
        self._map = index_map
        self._shape = tuple(shape)
        self._dtype = source.dtype
        self._nnz = nnz
    
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
    
    def row_values(self, i: int) -> Array:
        phys_row = int(self._map[i])
        return self._source.row_values(phys_row)
    
    def row_indices(self, i: int) -> Array:
        phys_row = int(self._map[i])
        return self._source.row_indices(phys_row)
    
    def row_length(self, i: int) -> int:
        phys_row = int(self._map[i])
        return self._source.row_length(phys_row)
    
    def sum(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        if axis is None:
            total = 0.0
            for i in range(self.rows):
                vals = self.row_values(i).to_numpy()
                total += np.sum(vals)
            return total
        elif axis == 1:
            result = np.zeros(self.rows, dtype=np.float64)
            for i in range(self.rows):
                vals = self.row_values(i).to_numpy()
                result[i] = np.sum(vals)
            return result
        else:
            result = np.zeros(self.cols, dtype=np.float64)
            for i in range(self.rows):
                vals = self.row_values(i).to_numpy()
                inds = self.row_indices(i).to_numpy()
                for k, j in enumerate(inds):
                    result[j] += vals[k]
            return result
    
    def mean(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        if axis is None:
            return self.sum() / self.size
        elif axis == 1:
            return self.sum(axis=1) / self.cols
        else:
            return self.sum(axis=0) / self.rows
    
    def var(self, axis: Optional[int] = None, ddof: int = 1) -> Union[float, np.ndarray]:
        return self.materialize().var(axis=axis, ddof=ddof)
    
    def materialize(self) -> 'CustomCSR':
        """Materialize to CustomCSR (deep copy)."""
        from ._custom import CustomCSR
        
        all_data, all_indices, indptr = [], [], [0]
        for i in range(self.rows):
            vals = self.row_values(i).to_numpy()
            inds = self.row_indices(i).to_numpy()
            all_data.extend(vals.tolist())
            all_indices.extend(inds.tolist())
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
        return self.materialize().to_scipy()
    
    def copy(self) -> 'CustomCSR':
        return self.materialize()
    
    @classmethod
    def _from_mapped_custom_slice(cls, source: 'MappedCustomCSR', row_indices: np.ndarray) -> 'MappedVirtualCSR':
        """Create MappedVirtualCSR from MappedCustomCSR row slice (INTERNAL)."""
        row_indices = np.asarray(row_indices, dtype=np.int64)
        total_nnz = sum(source.row_length(int(r)) for r in row_indices)
        return cls(source=source, index_map=row_indices, shape=(len(row_indices), source.cols),
                   nnz=total_nnz, _internal_key=_MAPPED_VIRTUAL_KEY)


# =============================================================================
# MappedVirtualCSC: Zero-Copy Column Slicing over Mapped Data
# =============================================================================

class MappedVirtualCSC(CSCBase):
    """
    Memory-mapped virtual CSC matrix - zero-copy view with index remapping.
    
    INTERNAL USE ONLY: Created by MappedCustomCSC.slice_cols().
    
    C++ Equivalent:
        scl::io::MappedVirtualSparse<Real, false>
    """
    
    __slots__ = ('_source', '_map', '_shape', '_dtype', '_nnz')
    
    def __init__(self, source: 'MappedCustomCSC', index_map: np.ndarray, shape: Tuple[int, int],
                 nnz: int, _internal_key=None):
        if _internal_key is not _MAPPED_VIRTUAL_KEY:
            raise TypeError("MappedVirtualCSC cannot be constructed directly. Use MappedCustomCSC.slice_cols().")
        self._source = source
        self._map = index_map
        self._shape = tuple(shape)
        self._dtype = source.dtype
        self._nnz = nnz
    
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
    
    def col_values(self, j: int) -> Array:
        return self._source.col_values(int(self._map[j]))
    
    def col_indices(self, j: int) -> Array:
        return self._source.col_indices(int(self._map[j]))
    
    def col_length(self, j: int) -> int:
        return self._source.col_length(int(self._map[j]))
    
    def sum(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        if axis is None:
            return sum(float(np.sum(self.col_values(j).to_numpy())) for j in range(self.cols))
        elif axis == 0:
            return np.array([np.sum(self.col_values(j).to_numpy()) for j in range(self.cols)])
        else:
            result = np.zeros(self.rows, dtype=np.float64)
            for j in range(self.cols):
                vals, inds = self.col_values(j).to_numpy(), self.col_indices(j).to_numpy()
                for k, i in enumerate(inds):
                    result[i] += vals[k]
            return result
    
    def mean(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        if axis is None:
            return self.sum() / self.size
        elif axis == 0:
            return self.sum(axis=0) / self.rows
        else:
            return self.sum(axis=1) / self.cols
    
    def var(self, axis: Optional[int] = None, ddof: int = 1) -> Union[float, np.ndarray]:
        return self.materialize().var(axis=axis, ddof=ddof)
    
    def materialize(self) -> 'CustomCSC':
        """Materialize to CustomCSC (deep copy)."""
        from ._custom import CustomCSC
        
        all_data, all_indices, indptr = [], [], [0]
        for j in range(self.cols):
            vals = self.col_values(j).to_numpy()
            inds = self.col_indices(j).to_numpy()
            all_data.extend(vals.tolist())
            all_indices.extend(inds.tolist())
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
        return self.materialize().to_scipy()
    
    def copy(self) -> 'CustomCSC':
        return self.materialize()
    
    @classmethod
    def _from_mapped_custom_slice(cls, source: 'MappedCustomCSC', col_indices: np.ndarray) -> 'MappedVirtualCSC':
        """Create MappedVirtualCSC from MappedCustomCSC column slice (INTERNAL)."""
        col_indices = np.asarray(col_indices, dtype=np.int64)
        total_nnz = sum(source.col_length(int(c)) for c in col_indices)
        return cls(source=source, index_map=col_indices, shape=(source.rows, len(col_indices)),
                   nnz=total_nnz, _internal_key=_MAPPED_VIRTUAL_KEY)
