"""
Lightweight Array Container

Pure Python/ctypes array implementation without numpy dependency.
Provides memory-aligned buffers compatible with SCL C API.
"""

import ctypes
from typing import Union, List, Any

from ._dtypes import DType

__all__ = ['Array', 'empty', 'zeros', 'from_list', 'from_buffer']


# =============================================================================
# Type Mapping
# =============================================================================

_TYPE_MAP = {
    'float32': (ctypes.c_float, 'f', 4),
    'float64': (ctypes.c_double, 'd', 8),
    'int32': (ctypes.c_int32, 'i', 4),
    'int64': (ctypes.c_int64, 'q', 8),
    'uint8': (ctypes.c_uint8, 'B', 1),
    'uint32': (ctypes.c_uint32, 'I', 4),
    'uint64': (ctypes.c_uint64, 'Q', 8),
}


def _get_type_info(dtype: str):
    """Get (ctypes_type, array_code, itemsize) for dtype string."""
    if dtype not in _TYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}. "
                        f"Supported: {list(_TYPE_MAP.keys())}")
    return _TYPE_MAP[dtype]


# =============================================================================
# Array Class
# =============================================================================

class Array:
    """
    Lightweight contiguous array with C-compatible memory layout.
    
    Features:
    - Memory-aligned allocation (64-byte for SIMD)
    - Zero-copy ctypes pointer access
    - Compatible with memoryview protocol
    - No numpy dependency
    
    Attributes:
        dtype (str): Data type ('float32', 'int64', etc.)
        size (int): Number of elements
        nbytes (int): Total bytes
        ptr (int): C pointer address (read-only)
    
    Example:
        >>> arr = Array.zeros(1000, dtype='float32')
        >>> arr[0] = 3.14
        >>> ptr = arr.get_pointer()  # For C API calls
        >>> view = memoryview(arr)   # Zero-copy buffer protocol
    """
    
    def __init__(
        self,
        size: int,
        dtype: Union[str, 'DType'] = 'float64',
        align: int = 64
    ):
        """
        Allocate uninitialized array.
        
        Args:
            size: Number of elements
            dtype: Data type (string or DType enum)
            align: Memory alignment in bytes (default: 64 for AVX-512)
        """
        if size < 0:
            raise ValueError(f"Array size must be non-negative, got {size}")
        
        # Normalize dtype (handle DType enum)
        if hasattr(dtype, 'value'):
            dtype = dtype.value
        
        self._size = size
        self._dtype = dtype
        self._align = align
        
        # Get type info
        self._ctype, self._array_code, self._itemsize = _get_type_info(dtype)
        self._nbytes = size * self._itemsize
        
        # Allocate aligned memory using ctypes
        if size == 0:
            self._data = None
            self._owner = None
        else:
            # Allocate with alignment
            # We allocate extra space for alignment
            extra = align + self._nbytes
            buffer = (ctypes.c_uint8 * extra)()
            
            # Calculate aligned address
            addr = ctypes.addressof(buffer)
            aligned_addr = (addr + align - 1) & ~(align - 1)
            offset = aligned_addr - addr
            
            # Create ctypes array at aligned address
            self._data = (self._ctype * size).from_address(aligned_addr)
            self._owner = buffer  # Keep reference to prevent GC
    
    @property
    def size(self) -> int:
        """Number of elements."""
        return self._size
    
    @property
    def dtype(self) -> str:
        """Data type string."""
        return self._dtype
    
    @property
    def nbytes(self) -> int:
        """Total bytes."""
        return self._nbytes
    
    @property
    def itemsize(self) -> int:
        """Bytes per element."""
        return self._itemsize
    
    @property
    def ptr(self) -> int:
        """C pointer address (read-only)."""
        if self._data is None:
            return 0
        return ctypes.addressof(self._data)
    
    def get_pointer(self) -> ctypes.c_void_p:
        """
        Get ctypes pointer for C API calls.
        
        Returns:
            ctypes pointer (c_void_p or typed pointer)
        """
        if self._data is None:
            return ctypes.c_void_p(0)
        return ctypes.cast(self._data, ctypes.c_void_p)
    
    def get_typed_pointer(self):
        """Get typed ctypes pointer (POINTER(c_float), etc.)."""
        if self._data is None:
            return ctypes.POINTER(self._ctype)()
        return ctypes.cast(self._data, ctypes.POINTER(self._ctype))
    
    # -------------------------------------------------------------------------
    # Initialization Methods
    # -------------------------------------------------------------------------
    
    @classmethod
    def zeros(cls, size: int, dtype: Union[str, 'DType'] = 'float64', align: int = 64) -> 'Array':
        """Create zero-initialized array."""
        arr = cls(size, dtype, align)
        if arr._data is not None:
            ctypes.memset(arr.get_pointer(), 0, arr.nbytes)
        return arr
    
    @classmethod
    def from_list(cls, data: List, dtype: Union[str, 'DType'] = 'float64', align: int = 64) -> 'Array':
        """Create array from Python list."""
        arr = cls(len(data), dtype, align)
        if arr._data is not None:
            for i, val in enumerate(data):
                arr._data[i] = val
        return arr
    
    @classmethod
    def from_buffer(cls, buffer: Any, dtype: Union[str, 'DType'], size: int) -> 'Array':
        """
        Create array from existing buffer (zero-copy view).
        
        WARNING: This creates a view, not a copy. The original buffer
        must remain alive while this Array exists.
        
        Args:
            buffer: Object supporting buffer protocol (bytes, memoryview, etc.)
            dtype: Data type string
            size: Number of elements
        """
        arr = cls.__new__(cls)
        arr._size = size
        
        # Normalize dtype
        if hasattr(dtype, 'value'):
            dtype = dtype.value
        
        arr._dtype = dtype
        arr._align = 0  # Not applicable for views
        
        ctype, _, itemsize = _get_type_info(dtype)
        arr._ctype = ctype
        arr._itemsize = itemsize
        arr._nbytes = size * itemsize
        
        # Create ctypes array from buffer
        try:
            mv = memoryview(buffer)
            if len(mv) < arr._nbytes:
                raise ValueError(f"Buffer too small: {len(mv)} < {arr._nbytes}")
            
            arr._data = (ctype * size).from_buffer(mv.obj if hasattr(mv, 'obj') else buffer)
            arr._owner = buffer  # Keep reference
        except Exception as e:
            raise TypeError(f"Cannot create Array from buffer: {e}")
        
        return arr
    
    # -------------------------------------------------------------------------
    # Element Access
    # -------------------------------------------------------------------------
    
    def __getitem__(self, idx: Union[int, slice]):
        """Get element(s) by index."""
        if self._data is None:
            raise IndexError("Empty array")
        
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._size)
            return [self._data[i] for i in range(start, stop, step or 1)]
        else:
            if idx < 0:
                idx += self._size
            if idx < 0 or idx >= self._size:
                raise IndexError(f"Index {idx} out of bounds [0, {self._size})")
            return self._data[idx]
    
    def __setitem__(self, idx: Union[int, slice], value):
        """Set element(s) by index."""
        if self._data is None:
            raise IndexError("Empty array")
        
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._size)
            indices = range(start, stop, step or 1)
            
            if hasattr(value, '__iter__'):
                for i, v in zip(indices, value):
                    self._data[i] = v
            else:
                for i in indices:
                    self._data[i] = value
        else:
            if idx < 0:
                idx += self._size
            if idx < 0 or idx >= self._size:
                raise IndexError(f"Index {idx} out of bounds [0, {self._size})")
            self._data[idx] = value
    
    def __len__(self) -> int:
        return self._size
    
    # -------------------------------------------------------------------------
    # Buffer Protocol
    # -------------------------------------------------------------------------
    
    def __buffer__(self, flags):
        """Support buffer protocol (Python 3.11+)."""
        if self._data is None:
            raise BufferError("Empty array has no buffer")
        return memoryview(self._data)
    
    def tobytes(self) -> bytes:
        """Convert to bytes."""
        if self._data is None:
            return b''
        return bytes(self._data)
    
    def tolist(self) -> List:
        """Convert to Python list."""
        if self._data is None:
            return []
        return list(self._data)
    
    # -------------------------------------------------------------------------
    # Copy Operations
    # -------------------------------------------------------------------------
    
    def copy(self) -> 'Array':
        """Create a deep copy."""
        new = Array(self._size, self._dtype, self._align)
        if self._data is not None and new._data is not None:
            ctypes.memmove(
                new.get_pointer(),
                self.get_pointer(),
                self.nbytes
            )
        return new
    
    def fill(self, value):
        """Fill array with a constant value."""
        if self._data is not None:
            for i in range(self._size):
                self._data[i] = value
    
    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        if self._size == 0:
            return f"Array([], dtype={self._dtype})"
        elif self._size <= 6:
            data_str = str(self.tolist())
        else:
            preview = self.tolist()[:3] + ['...'] + self.tolist()[-3:]
            data_str = str(preview)
        
        return f"Array({data_str}, dtype={self._dtype})"
    
    def __str__(self) -> str:
        return self.__repr__()


# =============================================================================
# Factory Functions
# =============================================================================

def empty(size: int, dtype: Union[str, 'DType'] = 'float64', align: int = 64) -> Array:
    """Create uninitialized array."""
    return Array(size, dtype, align)


def zeros(size: int, dtype: Union[str, 'DType'] = 'float64', align: int = 64) -> Array:
    """Create zero-initialized array."""
    return Array.zeros(size, dtype, align)


def from_list(data: List, dtype: Union[str, 'DType'] = 'float64', align: int = 64) -> Array:
    """Create array from Python list."""
    return Array.from_list(data, dtype, align)


def from_buffer(buffer: Any, dtype: Union[str, 'DType'], size: int) -> Array:
    """Create array from existing buffer (zero-copy)."""
    return Array.from_buffer(buffer, dtype, size)

