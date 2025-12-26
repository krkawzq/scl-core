"""
Data Type Definitions

Provides type-safe dtype constants and validation.
"""

from typing import Any, Union
from enum import Enum

__all__ = ['DType', 'float32', 'float64', 'int32', 'int64', 'uint8', 'uint32', 'uint64']


class DType(Enum):
    """
    SCL Data Type Enumeration.
    
    Provides type-safe constants for array creation.
    
    Example:
        >>> from scl.sparse import DType, Array
        >>> arr = Array.zeros(100, dtype=DType.float32)
        >>> 
        >>> # Or use module-level constants
        >>> import scl.sparse as sp
        >>> arr = Array.zeros(100, dtype=sp.float32)
    """
    
    float32 = 'float32'
    float64 = 'float64'
    int32 = 'int32'
    int64 = 'int64'
    uint8 = 'uint8'
    uint32 = 'uint32'
    uint64 = 'uint64'
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"DType.{self.name}"


# =============================================================================
# Module-Level Constants (For Clean Syntax)
# =============================================================================

float32 = DType.float32
float64 = DType.float64
int32 = DType.int32
int64 = DType.int64
uint8 = DType.uint8
uint32 = DType.uint32
uint64 = DType.uint64


# =============================================================================
# Type Utilities
# =============================================================================

def normalize_dtype(dtype: Union[str, DType]) -> str:
    """
    Normalize dtype to string.
    
    Args:
        dtype: String or DType enum
        
    Returns:
        String dtype
    
    Example:
        >>> normalize_dtype(DType.float32)
        'float32'
        >>> normalize_dtype('float64')
        'float64'
    """
    if isinstance(dtype, DType):
        return dtype.value
    elif isinstance(dtype, str):
        return dtype
    else:
        raise TypeError(f"dtype must be str or DType, got {type(dtype)}")


def validate_dtype(dtype: str) -> None:
    """
    Validate dtype string.
    
    Args:
        dtype: Data type string
        
    Raises:
        ValueError: If dtype is not supported
    """
    valid = {e.value for e in DType}
    if dtype not in valid:
        raise ValueError(f"Invalid dtype: {dtype}. Valid: {valid}")


def is_float_dtype(dtype: Union[str, DType]) -> bool:
    """Check if dtype is floating point."""
    dtype_str = normalize_dtype(dtype)
    return dtype_str in ('float32', 'float64')


def is_int_dtype(dtype: Union[str, DType]) -> bool:
    """Check if dtype is integer."""
    dtype_str = normalize_dtype(dtype)
    return dtype_str in ('int32', 'int64', 'uint8', 'uint32', 'uint64')


def dtype_itemsize(dtype: Union[str, DType]) -> int:
    """
    Get size in bytes for dtype.
    
    Args:
        dtype: Data type
        
    Returns:
        Size in bytes
    """
    dtype_str = normalize_dtype(dtype)
    
    size_map = {
        'float32': 4,
        'float64': 8,
        'int32': 4,
        'int64': 8,
        'uint8': 1,
        'uint32': 4,
        'uint64': 8,
    }
    
    return size_map[dtype_str]

