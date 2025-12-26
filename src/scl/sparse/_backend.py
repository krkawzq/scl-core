"""Backend Types and Storage Abstraction.

This module defines the backend system for sparse matrices, providing:
- Backend types (Custom, Virtual, Mapped)
- Storage abstraction for different data sources
- Unified interface for heterogeneous data

Design Philosophy:
    The backend system allows a single matrix class to transparently
    manage different underlying storage mechanisms. Users don't need
    to worry about whether data is owned, borrowed, or memory-mapped.

Backend Types:
    - CUSTOM: Locally owned data with full control
    - VIRTUAL: Zero-copy view supporting vstack and slicing
    - MAPPED: Memory-mapped file for out-of-core processing

Example:
    >>> mat = SclCSR.from_scipy(scipy_mat)  # CUSTOM backend, borrowed
    >>> view = mat[::2, :]                   # VIRTUAL backend, row slice
    >>> loaded = SclCSR.from_h5ad("data.h5ad")  # MAPPED backend
"""

from enum import Enum, auto
from typing import Any, Optional, Tuple, List, Union, TYPE_CHECKING
from dataclasses import dataclass, field
import weakref

if TYPE_CHECKING:
    from ._array import Array

__all__ = [
    'Backend',
    'Ownership', 
    'StorageInfo',
    'CustomStorage',
    'VirtualStorage',
    'MappedStorage',
    'ChunkInfo',
]


# =============================================================================
# Enumerations
# =============================================================================

class Backend(Enum):
    """Sparse matrix backend type.
    
    Determines how matrix data is stored and accessed.
    
    Attributes:
        CUSTOM: Data stored in local Arrays with direct ownership or borrowing.
                Supports all operations, optimal for small-medium matrices.
                
        VIRTUAL: Zero-copy view of one or more source matrices.
                 Supports logical vstack and row slicing without data copy.
                 Automatically flattens nested virtual matrices.
                 
        MAPPED: Memory-mapped file backend for large datasets.
                Data stays on disk, loaded on demand.
                Must be converted to CUSTOM or VIRTUAL for operations.
    
    Example:
        >>> mat.backend  # Backend.CUSTOM
        >>> view.backend  # Backend.VIRTUAL
    """
    CUSTOM = 'custom'
    VIRTUAL = 'virtual'
    MAPPED = 'mapped'


class Ownership(Enum):
    """Data ownership model.
    
    Determines memory management responsibility.
    
    Attributes:
        OWNED: Matrix owns the underlying arrays.
               Arrays will be freed when matrix is garbage collected.
               Created by: from_dense(), copy(), materialize()
               
        BORROWED: Matrix borrows external arrays.
                  Original owner must keep data alive.
                  Created by: from_scipy(), from_numpy()
                  
        VIEW: Matrix is a view into another matrix.
              Source matrix reference is maintained.
              Created by: slicing operations, vstack
    
    Memory Safety:
        - OWNED: Safe, no external dependencies
        - BORROWED: Caller must ensure source outlives matrix
        - VIEW: Automatically maintains reference chain
        
    Example:
        >>> owned = SclCSR.from_dense([[1, 2]])  # OWNED
        >>> borrowed = SclCSR.from_scipy(sp_mat)  # BORROWED
        >>> view = owned[0:10, :]  # VIEW
    """
    OWNED = 'owned'
    BORROWED = 'borrowed'
    VIEW = 'view'


# =============================================================================
# Storage Information
# =============================================================================

@dataclass
class StorageInfo:
    """Storage metadata for a sparse matrix.
    
    Contains all information about how matrix data is stored,
    including backend type, ownership, and source references.
    
    Attributes:
        backend: Storage backend type.
        ownership: Data ownership model.
        dtype: Data type string ('float32', 'float64').
        shape: Matrix dimensions (rows, cols).
        nnz: Number of non-zero elements (may be approximate for VIRTUAL).
        
    Note:
        This is primarily for introspection and debugging.
        Normal usage doesn't require accessing this directly.
    """
    backend: Backend
    ownership: Ownership
    dtype: str
    shape: Tuple[int, int]
    nnz: int
    
    # Optional metadata
    source_file: Optional[str] = None  # For MAPPED backend
    is_contiguous: bool = True  # False for sliced views
    
    def __repr__(self) -> str:
        return (
            f"StorageInfo(backend={self.backend.value}, "
            f"ownership={self.ownership.value}, "
            f"dtype={self.dtype}, shape={self.shape}, nnz={self.nnz})"
        )


# =============================================================================
# Custom Storage (Owned/Borrowed Arrays)
# =============================================================================

@dataclass
class CustomStorage:
    """Storage for CUSTOM backend matrices.
    
    Holds the actual sparse matrix arrays (data, indices, indptr)
    with ownership tracking.
    
    Attributes:
        data: Non-zero values array.
        indices: Index array (col indices for CSR, row indices for CSC).
        indptr: Index pointer array.
        primary_lengths: Cached lengths per primary axis element.
        ownership: Whether data is owned or borrowed.
        _source_ref: Weak reference to borrowed source (if any).
        
    Memory Model:
        - OWNED: Arrays are scl.sparse.Array instances we created
        - BORROWED: Arrays wrap external memory (numpy/scipy)
        
    Example:
        >>> storage = CustomStorage(data, indices, indptr, lengths, Ownership.OWNED)
        >>> storage.is_owned  # True
    """
    data: 'Array'
    indices: 'Array'
    indptr: 'Array'
    primary_lengths: Optional['Array'] = None
    ownership: Ownership = Ownership.OWNED
    
    # For borrowed data, keep reference to source
    _source_ref: Any = field(default=None, repr=False)
    
    @property
    def is_owned(self) -> bool:
        """Check if storage owns its data."""
        return self.ownership == Ownership.OWNED
    
    @property
    def is_borrowed(self) -> bool:
        """Check if storage borrows external data."""
        return self.ownership == Ownership.BORROWED
    
    @property
    def source(self) -> Optional[Any]:
        """Get borrowed source (if any)."""
        if self._source_ref is None:
            return None
        if isinstance(self._source_ref, weakref.ref):
            return self._source_ref()
        return self._source_ref
    
    def get_pointers(self) -> Tuple:
        """Get C-compatible pointers.
        
        Returns:
            (data_ptr, indices_ptr, indptr_ptr, lengths_ptr)
        """
        return (
            self.data.get_pointer(),
            self.indices.get_pointer(),
            self.indptr.get_pointer(),
            self.primary_lengths.get_pointer() if self.primary_lengths else None
        )
    
    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.data)
    
    @property
    def nbytes(self) -> int:
        """Total memory usage in bytes."""
        total = self.data.nbytes + self.indices.nbytes + self.indptr.nbytes
        if self.primary_lengths:
            total += self.primary_lengths.nbytes
        return total


# =============================================================================
# Virtual Storage (Zero-Copy Views)
# =============================================================================

@dataclass
class ChunkInfo:
    """Information about a single chunk in virtual storage.
    
    A virtual matrix is composed of one or more chunks, each
    representing a contiguous block from a source matrix.
    
    Attributes:
        source: Reference to source matrix (strong ref to prevent GC).
        local_indices: Row/col indices within source (None = all).
        offset: Global offset of this chunk in virtual matrix.
        length: Number of rows/cols in this chunk.
        
    Note:
        Strong references are used intentionally to prevent
        source matrices from being garbage collected.
    """
    source: Any  # Strong reference to source matrix
    local_indices: Optional['Array'] = None  # None means all rows/cols
    offset: int = 0
    length: int = 0
    
    @property
    def is_identity(self) -> bool:
        """Check if chunk uses identity mapping (no indirection)."""
        return self.local_indices is None
    
    @property 
    def is_sliced(self) -> bool:
        """Check if chunk is a slice (has indirection)."""
        return self.local_indices is not None


@dataclass
class VirtualStorage:
    """Storage for VIRTUAL backend matrices.
    
    Represents a logical view over one or more source matrices,
    supporting zero-copy vstack and row slicing operations.
    
    Attributes:
        chunks: List of chunk information.
        total_primary: Total size along primary axis.
        secondary_size: Size along secondary axis (must match all chunks).
        dtype: Data type (must match all chunks).
        
    Design:
        Virtual storage maintains strong references to all source
        matrices to prevent use-after-free bugs. When a virtual
        matrix is created from another virtual, the chunk list
        is flattened to avoid deep nesting.
        
    Example:
        >>> # vstack creates virtual storage
        >>> stacked = vstack_csr([mat1, mat2])
        >>> stacked._storage.chunks  # [ChunkInfo(mat1), ChunkInfo(mat2)]
        
        >>> # Slicing creates virtual with local_indices
        >>> sliced = mat[::2, :]
        >>> sliced._storage.chunks[0].local_indices  # [0, 2, 4, ...]
    """
    chunks: List[ChunkInfo] = field(default_factory=list)
    total_primary: int = 0
    secondary_size: int = 0
    dtype: str = 'float64'
    
    @property
    def n_chunks(self) -> int:
        """Number of chunks."""
        return len(self.chunks)
    
    @property
    def is_single_chunk(self) -> bool:
        """Check if virtual has only one chunk."""
        return len(self.chunks) == 1
    
    @property
    def is_contiguous(self) -> bool:
        """Check if virtual is single chunk with identity mapping."""
        return self.is_single_chunk and self.chunks[0].is_identity
    
    def get_chunk_for_index(self, idx: int) -> Tuple[int, int]:
        """Find chunk containing global index.
        
        Args:
            idx: Global index along primary axis.
            
        Returns:
            (chunk_index, local_index_within_chunk)
        """
        current = 0
        for i, chunk in enumerate(self.chunks):
            if current <= idx < current + chunk.length:
                return i, idx - current
            current += chunk.length
        raise IndexError(f"Index {idx} out of bounds [0, {self.total_primary})")
    
    @property
    def nnz(self) -> int:
        """Compute total NNZ (may require inspection)."""
        total = 0
        for chunk in self.chunks:
            if chunk.is_identity:
                total += chunk.source.nnz
            else:
                # Need to sum lengths of selected rows/cols
                src = chunk.source
                if hasattr(src, 'primary_lengths'):
                    for idx in chunk.local_indices.tolist():
                        total += src.primary_lengths[idx]
                else:
                    # Fallback: use indptr
                    for idx in chunk.local_indices.tolist():
                        total += src.indptr[idx + 1] - src.indptr[idx]
        return total
    
    def flatten(self) -> 'VirtualStorage':
        """Flatten nested virtual storage.
        
        If any chunk's source is also virtual, expand it inline
        to avoid deep nesting.
        
        Returns:
            Flattened VirtualStorage (may be self if already flat).
        """
        new_chunks = []
        needs_flatten = False
        
        for chunk in self.chunks:
            src = chunk.source
            
            # Check if source is virtual
            if hasattr(src, '_storage') and isinstance(src._storage, VirtualStorage):
                needs_flatten = True
                vstorage = src._storage
                
                # Expand source's chunks
                for src_chunk in vstorage.chunks:
                    if chunk.is_identity:
                        # Pass through source chunk as-is
                        new_chunks.append(ChunkInfo(
                            source=src_chunk.source,
                            local_indices=src_chunk.local_indices,
                            offset=len(new_chunks),
                            length=src_chunk.length
                        ))
                    else:
                        # Compose indices
                        if src_chunk.is_identity:
                            # Apply our indices to source
                            new_chunks.append(ChunkInfo(
                                source=src_chunk.source,
                                local_indices=chunk.local_indices,
                                offset=len(new_chunks),
                                length=chunk.length
                            ))
                        else:
                            # Compose: our_indices[src_indices]
                            from ._array import empty
                            composed = empty(len(chunk.local_indices), dtype='int64')
                            src_indices = src_chunk.local_indices.tolist()
                            for i, our_idx in enumerate(chunk.local_indices.tolist()):
                                composed[i] = src_indices[our_idx]
                            new_chunks.append(ChunkInfo(
                                source=src_chunk.source,
                                local_indices=composed,
                                offset=len(new_chunks),
                                length=chunk.length
                            ))
            else:
                new_chunks.append(chunk)
        
        if not needs_flatten:
            return self
        
        return VirtualStorage(
            chunks=new_chunks,
            total_primary=self.total_primary,
            secondary_size=self.secondary_size,
            dtype=self.dtype
        )


# =============================================================================
# Mapped Storage (Memory-Mapped Files)
# =============================================================================

@dataclass
class MappedStorage:
    """Storage for MAPPED backend matrices.
    
    Represents a sparse matrix stored in a file (HDF5, Zarr, etc.)
    with lazy loading capabilities.
    
    Attributes:
        path: Path to the data file.
        format: File format ('h5ad', 'zarr', 'npz', etc.).
        group: Group/key within file (for HDF5/Zarr).
        shape: Matrix shape (known from metadata).
        nnz: Number of non-zeros (known from metadata).
        dtype: Data type.
        _handle: Open file handle (lazy).
        
    Usage:
        Mapped storage cannot be used directly for computation.
        Use load() to convert to CUSTOM, or load_virtual() for
        partial loading.
        
    Example:
        >>> mat = SclCSR.from_h5ad("data.h5ad", backed=True)
        >>> mat.backend  # Backend.MAPPED
        >>> mat.shape  # Known from metadata
        >>> 
        >>> # Load fully
        >>> loaded = mat.load()  # Backend.CUSTOM
        >>> 
        >>> # Load subset
        >>> subset = mat[0:1000, :].load()  # Only load 1000 rows
    """
    path: str
    format: str = 'h5ad'
    group: str = 'X'
    shape: Tuple[int, int] = (0, 0)
    nnz: int = 0
    dtype: str = 'float64'
    
    # Lazy file handle
    _handle: Any = field(default=None, repr=False)
    
    @property
    def is_open(self) -> bool:
        """Check if file handle is open."""
        return self._handle is not None
    
    def open(self) -> None:
        """Open file handle."""
        if self._handle is not None:
            return
            
        if self.format == 'h5ad':
            try:
                import h5py
                self._handle = h5py.File(self.path, 'r')
            except ImportError:
                raise ImportError("h5py required for HDF5 files")
        elif self.format == 'zarr':
            try:
                import zarr
                self._handle = zarr.open(self.path, 'r')
            except ImportError:
                raise ImportError("zarr required for Zarr files")
        else:
            raise ValueError(f"Unknown format: {self.format}")
    
    def close(self) -> None:
        """Close file handle."""
        if self._handle is not None:
            if hasattr(self._handle, 'close'):
                self._handle.close()
            self._handle = None
    
    def __del__(self):
        """Cleanup on garbage collection."""
        self.close()

