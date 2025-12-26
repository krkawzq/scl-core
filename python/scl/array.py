"""
SCL Array - Zero-dependency Array Implementation

Provides memory-managed arrays using SCL's memory allocator.
No external dependencies (no numpy).

Types:
    - Real: double (float64)
    - Index: int64
    - Byte: uint8

Usage:
    arr = RealArray(100)       # Allocate 100 doubles
    arr[0] = 1.5               # Set value
    arr[0]                     # Get value -> 1.5
    arr.size                   # -> 100
    arr.data_ptr               # -> raw pointer for C API
"""

from __future__ import annotations

import ctypes
from ctypes import c_double, c_int64, c_uint8, c_void_p, byref, POINTER, cast
from typing import Union, Iterator, Sequence, Optional, Type

from scl._ffi import get_lib_with_signatures, check_error
from scl._dtypes import (
    DType, SIZEOF_REAL, SIZEOF_INDEX, SIZEOF_BYTE,
    RealCType, IndexCType, ByteCType, SCL_ALIGNMENT,
)


# =============================================================================
# Base Array Class
# =============================================================================

class ArrayBase:
    """
    Base class for SCL arrays.

    Manages memory allocation using SCL's allocator (scl_malloc/scl_free).
    """

    __slots__ = ("_ptr", "_size", "_owns_memory")

    # Subclasses must define these
    _ctype = None       # ctypes type (c_double, c_int64, c_uint8)
    _sizeof = 0         # sizeof element

    def __init__(self, size: int = 0, *, _ptr: Optional[c_void_p] = None,
                 _owns_memory: bool = True):
        """
        Initialize array.

        Args:
            size: Number of elements
            _ptr: Optional existing pointer (for wrapping)
            _owns_memory: If False, don't free memory on destruction
        """
        self._size = size
        self._owns_memory = _owns_memory

        if _ptr is not None:
            self._ptr = _ptr
        elif size > 0:
            lib = get_lib_with_signatures()
            ptr = c_void_p()
            check_error(lib.scl_calloc(size * self._sizeof, byref(ptr)))
            self._ptr = ptr
        else:
            self._ptr = c_void_p()

    def __del__(self):
        self._release()

    def _release(self):
        """Release memory if owned."""
        if self._owns_memory and self._ptr and self._ptr.value:
            try:
                lib = get_lib_with_signatures()
                lib.scl_free(self._ptr)
            except Exception:
                pass
            self._ptr = c_void_p()

    @property
    def size(self) -> int:
        """Number of elements."""
        return self._size

    @property
    def nbytes(self) -> int:
        """Total bytes."""
        return self._size * self._sizeof

    @property
    def data_ptr(self) -> int:
        """Raw pointer address for C API."""
        return self._ptr.value or 0

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def _check_index(self, i: int) -> int:
        """Validate and normalize index."""
        if i < 0:
            i += self._size
        if i < 0 or i >= self._size:
            raise IndexError(f"Index {i} out of range [0, {self._size})")
        return i

    def _get_typed_ptr(self):
        """Get pointer cast to correct type."""
        return cast(self._ptr, POINTER(self._ctype))

    def __getitem__(self, i: int):
        """Get element at index."""
        i = self._check_index(i)
        return self._get_typed_ptr()[i]

    def __setitem__(self, i: int, value):
        """Set element at index."""
        i = self._check_index(i)
        self._get_typed_ptr()[i] = value

    def __iter__(self) -> Iterator:
        """Iterate over elements."""
        ptr = self._get_typed_ptr()
        for i in range(self._size):
            yield ptr[i]

    def fill(self, value) -> "ArrayBase":
        """Fill array with value."""
        ptr = self._get_typed_ptr()
        for i in range(self._size):
            ptr[i] = value
        return self

    def zero(self) -> "ArrayBase":
        """Zero the array."""
        if self._ptr and self._ptr.value:
            lib = get_lib_with_signatures()
            lib.scl_memzero(self._ptr, self.nbytes)
        return self

    def copy(self) -> "ArrayBase":
        """Create a copy."""
        new_arr = type(self)(self._size)
        if self._size > 0:
            lib = get_lib_with_signatures()
            check_error(lib.scl_memcpy(self._ptr, new_arr._ptr, self.nbytes))
        return new_arr

    def to_list(self) -> list:
        """Convert to Python list."""
        return list(self)

    def __repr__(self) -> str:
        name = type(self).__name__
        if self._size <= 8:
            content = ", ".join(str(x) for x in self)
            return f"{name}([{content}])"
        else:
            first = ", ".join(str(self[i]) for i in range(4))
            last = ", ".join(str(self[i]) for i in range(self._size - 2, self._size))
            return f"{name}([{first}, ..., {last}], size={self._size})"


# =============================================================================
# Typed Array Classes
# =============================================================================

class RealArray(ArrayBase):
    """
    Array of Real (double/float64) values.

    This is the primary array type for data values in SCL.
    """

    _ctype = RealCType
    _sizeof = SIZEOF_REAL
    _dtype = DType.FLOAT64

    @classmethod
    def from_sequence(cls, seq: Sequence[float]) -> "RealArray":
        """Create from Python sequence."""
        arr = cls(len(seq))
        ptr = arr._get_typed_ptr()
        for i, v in enumerate(seq):
            ptr[i] = float(v)
        return arr

    @classmethod
    def wrap(cls, ptr: int, size: int) -> "RealArray":
        """
        Wrap existing pointer (non-owning).

        Warning: Caller must ensure pointer remains valid.
        """
        return cls(size, _ptr=c_void_p(ptr), _owns_memory=False)


class IndexArray(ArrayBase):
    """
    Array of Index (int64) values.

    Used for indices, indptr in sparse matrices.
    """

    _ctype = IndexCType
    _sizeof = SIZEOF_INDEX
    _dtype = DType.INT64

    @classmethod
    def from_sequence(cls, seq: Sequence[int]) -> "IndexArray":
        """Create from Python sequence."""
        arr = cls(len(seq))
        ptr = arr._get_typed_ptr()
        for i, v in enumerate(seq):
            ptr[i] = int(v)
        return arr

    @classmethod
    def wrap(cls, ptr: int, size: int) -> "IndexArray":
        """Wrap existing pointer (non-owning)."""
        return cls(size, _ptr=c_void_p(ptr), _owns_memory=False)


class ByteArray(ArrayBase):
    """
    Array of Byte (uint8) values.

    Used for masks and flags.
    """

    _ctype = ByteCType
    _sizeof = SIZEOF_BYTE
    _dtype = DType.UINT8

    @classmethod
    def from_sequence(cls, seq: Sequence[int]) -> "ByteArray":
        """Create from Python sequence (values 0-255)."""
        arr = cls(len(seq))
        ptr = arr._get_typed_ptr()
        for i, v in enumerate(seq):
            ptr[i] = int(v) & 0xFF
        return arr

    @classmethod
    def from_mask(cls, mask: Sequence[bool]) -> "ByteArray":
        """Create from boolean sequence."""
        arr = cls(len(mask))
        ptr = arr._get_typed_ptr()
        for i, v in enumerate(mask):
            ptr[i] = 1 if v else 0
        return arr

    @classmethod
    def wrap(cls, ptr: int, size: int) -> "ByteArray":
        """Wrap existing pointer (non-owning)."""
        return cls(size, _ptr=c_void_p(ptr), _owns_memory=False)

    def count_nonzero(self) -> int:
        """Count non-zero elements."""
        count = 0
        ptr = self._get_typed_ptr()
        for i in range(self._size):
            if ptr[i]:
                count += 1
        return count


# =============================================================================
# Utility Functions
# =============================================================================

def zeros_real(size: int) -> RealArray:
    """Create zero-initialized RealArray."""
    return RealArray(size)


def zeros_index(size: int) -> IndexArray:
    """Create zero-initialized IndexArray."""
    return IndexArray(size)


def zeros_byte(size: int) -> ByteArray:
    """Create zero-initialized ByteArray."""
    return ByteArray(size)


def ones_byte(size: int) -> ByteArray:
    """Create ByteArray filled with 1s."""
    return ByteArray(size).fill(1)


# =============================================================================
# Numpy Interop (Optional)
# =============================================================================

def from_numpy(arr) -> Union[RealArray, IndexArray, ByteArray]:
    """
    Create SCL array from numpy array (copies data).

    Supported dtypes: float64, int64, uint8
    """
    import numpy as np

    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    if arr.dtype == np.float64:
        scl_arr = RealArray(arr.size)
    elif arr.dtype == np.int64:
        scl_arr = IndexArray(arr.size)
    elif arr.dtype == np.uint8:
        scl_arr = ByteArray(arr.size)
    else:
        raise TypeError(f"Unsupported dtype: {arr.dtype}")

    # Copy data
    lib = get_lib_with_signatures()
    check_error(lib.scl_memcpy(
        ctypes.c_void_p(arr.ctypes.data),
        scl_arr._ptr,
        scl_arr.nbytes
    ))

    return scl_arr


def to_numpy(arr: ArrayBase):
    """
    Create numpy array from SCL array (copies data).
    """
    import numpy as np

    if isinstance(arr, RealArray):
        dtype = np.float64
    elif isinstance(arr, IndexArray):
        dtype = np.int64
    elif isinstance(arr, ByteArray):
        dtype = np.uint8
    else:
        raise TypeError(f"Unknown array type: {type(arr)}")

    np_arr = np.empty(arr.size, dtype=dtype)

    lib = get_lib_with_signatures()
    check_error(lib.scl_memcpy(
        arr._ptr,
        ctypes.c_void_p(np_arr.ctypes.data),
        arr.nbytes
    ))

    return np_arr


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants (re-exported from _dtypes)
    "SIZEOF_REAL",
    "SIZEOF_INDEX",
    "SIZEOF_BYTE",
    # DType (re-exported)
    "DType",
    # Classes
    "ArrayBase",
    "RealArray",
    "IndexArray",
    "ByteArray",
    # Factory functions
    "zeros_real",
    "zeros_index",
    "zeros_byte",
    "ones_byte",
    # Numpy interop
    "from_numpy",
    "to_numpy",
]

# Re-export DType for convenience
from scl._dtypes import DType
