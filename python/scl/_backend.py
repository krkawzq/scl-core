"""
SCL Backend - Storage Backend Abstraction

Provides abstraction for different storage backends:
- Custom: Data owned by SCL or borrowed from external source
- Virtual: Zero-copy view composed from multiple chunks
- Mapped: Memory-mapped file storage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Tuple, Callable, Any, TYPE_CHECKING
import weakref

if TYPE_CHECKING:
    from scl.array import RealArray, IndexArray


# =============================================================================
# Backend Enumeration
# =============================================================================

class Backend(IntEnum):
    """
    Storage backend type.
    """
    CUSTOM = 0      # In-memory (owned or borrowed)
    VIRTUAL = 1     # Virtual matrix (composed from chunks)
    MAPPED = 2      # Memory-mapped file


class Ownership(IntEnum):
    """
    Memory ownership status.
    """
    OWNED = 0       # SCL owns the memory (will free on destruction)
    BORROWED = 1    # Memory borrowed from external source (no free)
    VIEW = 2        # View of another matrix (reference counted)


class SourceType(IntEnum):
    """
    Source type for virtual matrices.
    """
    PARENT = 0      # Direct parent reference
    SLICE = 1       # Sliced from parent
    CONCAT = 2      # Concatenated from multiple sources
    REORDER = 3     # Reordered view


# =============================================================================
# Storage Information
# =============================================================================

@dataclass
class StorageInfo:
    """
    Base storage information.

    Tracks backend type, ownership, and memory usage.
    """
    backend: Backend = Backend.CUSTOM
    ownership: Ownership = Ownership.OWNED
    memory_bytes: int = 0
    is_contiguous: bool = True

    @property
    def is_owned(self) -> bool:
        """Whether data is owned by this matrix."""
        return self.ownership == Ownership.OWNED

    @property
    def is_view(self) -> bool:
        """Whether this is a view."""
        return self.ownership == Ownership.VIEW

    @property
    def is_borrowed(self) -> bool:
        """Whether memory is borrowed."""
        return self.ownership == Ownership.BORROWED


@dataclass
class CustomStorage(StorageInfo):
    """
    Storage info for custom (in-memory) backend.

    Custom storage means data is directly in memory, either:
    - OWNED: Allocated by SCL, will be freed on destruction
    - BORROWED: Provided by user, SCL will not free
    """
    backend: Backend = field(default=Backend.CUSTOM, init=False)
    source_ptr: int = 0  # Original pointer (for borrowed data)

    def __post_init__(self):
        self.backend = Backend.CUSTOM


@dataclass
class ChunkInfo:
    """
    Information about a chunk in virtual matrix.

    Virtual matrices are composed of multiple chunks that
    can be from different sources.
    """
    source: Any = None           # Weak reference to source matrix
    row_start: int = 0           # Start row in source
    row_end: int = 0             # End row in source
    col_start: int = 0           # Start column in source
    col_end: int = 0             # End column in source
    is_transposed: bool = False  # Whether this chunk is transposed

    @property
    def num_rows(self) -> int:
        """Number of rows in chunk."""
        return self.row_end - self.row_start

    @property
    def num_cols(self) -> int:
        """Number of columns in chunk."""
        return self.col_end - self.col_start

    def get_source(self):
        """Get source matrix (resolving weak reference)."""
        if self.source is None:
            return None
        if isinstance(self.source, weakref.ref):
            return self.source()
        return self.source


@dataclass
class VirtualStorage(StorageInfo):
    """
    Storage info for virtual (zero-copy) backend.

    Virtual matrices are composed from chunks without copying data.
    The chunks maintain weak references to their sources.
    """
    backend: Backend = field(default=Backend.VIRTUAL, init=False)
    ownership: Ownership = field(default=Ownership.VIEW, init=False)
    source_type: SourceType = SourceType.PARENT
    chunks: List[ChunkInfo] = field(default_factory=list)

    # Lazy evaluation state
    row_mask: Optional[Any] = None      # ByteArray for row selection
    col_mask: Optional[Any] = None      # ByteArray for column selection
    row_order: Optional[Any] = None     # IndexArray for row reordering
    col_order: Optional[Any] = None     # IndexArray for column reordering

    def __post_init__(self):
        self.backend = Backend.VIRTUAL
        self.ownership = Ownership.VIEW
        self.is_contiguous = False

    def add_chunk(self, source: Any, row_range: Tuple[int, int],
                  col_range: Tuple[int, int], transposed: bool = False):
        """Add a chunk to this virtual matrix."""
        chunk = ChunkInfo(
            source=weakref.ref(source) if source is not None else None,
            row_start=row_range[0],
            row_end=row_range[1],
            col_start=col_range[0],
            col_end=col_range[1],
            is_transposed=transposed,
        )
        self.chunks.append(chunk)

    def has_pending_ops(self) -> bool:
        """Check if there are pending lazy operations."""
        return (self.row_mask is not None or
                self.col_mask is not None or
                self.row_order is not None or
                self.col_order is not None)


@dataclass
class MappedStorage(StorageInfo):
    """
    Storage info for memory-mapped file backend.

    Mapped storage uses memory-mapped files for efficient
    access to large datasets that don't fit in memory.
    """
    backend: Backend = field(default=Backend.MAPPED, init=False)
    ownership: Ownership = field(default=Ownership.VIEW, init=False)

    file_path: str = ""
    handle: int = 0              # C API handle
    max_pages: int = 64          # Maximum resident pages
    page_size: int = 4096        # Page size in bytes

    # File layout info
    header_size: int = 0
    data_offset: int = 0
    indices_offset: int = 0
    indptr_offset: int = 0

    def __post_init__(self):
        self.backend = Backend.MAPPED
        self.ownership = Ownership.VIEW
        self.is_contiguous = False


# =============================================================================
# Backend Factory
# =============================================================================

class BackendFactory:
    """
    Factory for creating storage backends.
    """

    @staticmethod
    def custom(ownership: Ownership = Ownership.OWNED,
               memory_bytes: int = 0,
               source_ptr: int = 0) -> CustomStorage:
        """Create custom storage info."""
        return CustomStorage(
            ownership=ownership,
            memory_bytes=memory_bytes,
            source_ptr=source_ptr,
        )

    @staticmethod
    def virtual(source_type: SourceType = SourceType.PARENT) -> VirtualStorage:
        """Create virtual storage info."""
        return VirtualStorage(source_type=source_type)

    @staticmethod
    def mapped(file_path: str, handle: int,
               max_pages: int = 64) -> MappedStorage:
        """Create mapped storage info."""
        return MappedStorage(
            file_path=file_path,
            handle=handle,
            max_pages=max_pages,
        )


# =============================================================================
# Backend Registry
# =============================================================================

class BackendRegistry:
    """
    Registry for backend-specific operations.

    Allows registration of custom handlers for different backends.
    """

    def __init__(self):
        self._handlers: dict[Backend, dict[str, Callable]] = {
            Backend.CUSTOM: {},
            Backend.VIRTUAL: {},
            Backend.MAPPED: {},
        }

    def register(self, backend: Backend, operation: str, handler: Callable):
        """Register a handler for an operation on a backend."""
        self._handlers[backend][operation] = handler

    def get_handler(self, backend: Backend, operation: str) -> Optional[Callable]:
        """Get handler for an operation."""
        return self._handlers[backend].get(operation)

    def has_handler(self, backend: Backend, operation: str) -> bool:
        """Check if handler exists."""
        return operation in self._handlers[backend]


# Global registry instance
_registry = BackendRegistry()


def register_handler(backend: Backend, operation: str):
    """Decorator to register a backend handler."""
    def decorator(func: Callable) -> Callable:
        _registry.register(backend, operation, func)
        return func
    return decorator


def get_handler(backend: Backend, operation: str) -> Optional[Callable]:
    """Get registered handler."""
    return _registry.get_handler(backend, operation)


# =============================================================================
# Backend Selection
# =============================================================================

def suggest_backend(data_bytes: int, available_mb: int = 4096) -> Backend:
    """
    Suggest optimal backend based on data size and available memory.

    Args:
        data_bytes: Total data size in bytes
        available_mb: Available memory in MB

    Returns:
        Suggested backend type
    """
    available_bytes = available_mb * 1024 * 1024

    # If data fits in 50% of available memory, use in-memory
    if data_bytes < available_bytes * 0.5:
        return Backend.CUSTOM

    # Otherwise, use memory-mapped
    return Backend.MAPPED


def estimate_memory(rows: int, nnz: int) -> int:
    """
    Estimate memory requirements for sparse matrix.

    Args:
        rows: Number of rows
        nnz: Number of non-zeros

    Returns:
        Estimated bytes
    """
    from scl._dtypes import SIZEOF_REAL, SIZEOF_INDEX

    # data array: nnz * sizeof(Real)
    # indices array: nnz * sizeof(Index)
    # indptr array: (rows + 1) * sizeof(Index)
    return (nnz * SIZEOF_REAL +
            nnz * SIZEOF_INDEX +
            (rows + 1) * SIZEOF_INDEX)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "Backend",
    "Ownership",
    "SourceType",
    # Storage classes
    "StorageInfo",
    "CustomStorage",
    "VirtualStorage",
    "MappedStorage",
    "ChunkInfo",
    # Factory
    "BackendFactory",
    # Registry
    "BackendRegistry",
    "register_handler",
    "get_handler",
    # Utilities
    "suggest_backend",
    "estimate_memory",
]
