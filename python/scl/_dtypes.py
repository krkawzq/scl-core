"""
SCL DTypes - Data Type Definitions

Zero external dependencies. Defines data types and type mapping
for the SCL sparse computing library.
"""

from __future__ import annotations

import ctypes
from ctypes import c_double, c_int64, c_uint8, c_float, c_int32
from enum import IntEnum
from typing import Type, Union, Optional, Dict, Any


# =============================================================================
# Data Type Enumeration
# =============================================================================

class DType(IntEnum):
    """
    Supported data types in SCL.

    Each type has corresponding C type, size, and alignment requirements.
    """
    FLOAT32 = 0    # 32-bit floating point (not primary, for compatibility)
    FLOAT64 = 1    # 64-bit floating point (Real)
    INT32 = 2      # 32-bit signed integer
    INT64 = 3      # 64-bit signed integer (Index)
    UINT8 = 4      # 8-bit unsigned integer (Byte/mask)
    BOOL = 5       # Boolean (stored as uint8)

    @property
    def itemsize(self) -> int:
        """Size in bytes of one element."""
        return _DTYPE_INFO[self]["size"]

    @property
    def ctype(self) -> Type:
        """Corresponding ctypes type."""
        return _DTYPE_INFO[self]["ctype"]

    @property
    def name(self) -> str:
        """Human-readable name."""
        return _DTYPE_INFO[self]["name"]

    @property
    def alignment(self) -> int:
        """Preferred alignment in bytes."""
        return _DTYPE_INFO[self]["alignment"]

    @classmethod
    def from_ctype(cls, ctype: Type) -> "DType":
        """Get DType from ctypes type."""
        for dtype, info in _DTYPE_INFO.items():
            if info["ctype"] == ctype:
                return dtype
        raise ValueError(f"Unknown ctype: {ctype}")

    @classmethod
    def from_name(cls, name: str) -> "DType":
        """Get DType from string name."""
        name_lower = name.lower()
        for dtype, info in _DTYPE_INFO.items():
            if info["name"].lower() == name_lower:
                return dtype
        # Aliases
        aliases = {
            "double": cls.FLOAT64,
            "float": cls.FLOAT32,
            "real": cls.FLOAT64,
            "index": cls.INT64,
            "byte": cls.UINT8,
            "int": cls.INT64,
            "long": cls.INT64,
        }
        if name_lower in aliases:
            return aliases[name_lower]
        raise ValueError(f"Unknown dtype name: {name}")


# Type information table
_DTYPE_INFO: Dict[DType, Dict[str, Any]] = {
    DType.FLOAT32: {
        "ctype": c_float,
        "size": 4,
        "alignment": 4,
        "name": "float32",
    },
    DType.FLOAT64: {
        "ctype": c_double,
        "size": 8,
        "alignment": 8,
        "name": "float64",
    },
    DType.INT32: {
        "ctype": c_int32,
        "size": 4,
        "alignment": 4,
        "name": "int32",
    },
    DType.INT64: {
        "ctype": c_int64,
        "size": 8,
        "alignment": 8,
        "name": "int64",
    },
    DType.UINT8: {
        "ctype": c_uint8,
        "size": 1,
        "alignment": 1,
        "name": "uint8",
    },
    DType.BOOL: {
        "ctype": c_uint8,
        "size": 1,
        "alignment": 1,
        "name": "bool",
    },
}


# =============================================================================
# Type Mapping (Python type -> DType)
# =============================================================================

TYPE_MAP: Dict[type, DType] = {
    float: DType.FLOAT64,
    int: DType.INT64,
    bool: DType.BOOL,
}

CTYPE_MAP: Dict[Type, DType] = {
    c_double: DType.FLOAT64,
    c_float: DType.FLOAT32,
    c_int64: DType.INT64,
    c_int32: DType.INT32,
    c_uint8: DType.UINT8,
}


# =============================================================================
# Primary SCL Types
# =============================================================================

# Primary types used in SCL
Real = DType.FLOAT64      # Primary floating-point type
Index = DType.INT64       # Primary integer type for indices
Byte = DType.UINT8        # Primary byte type for masks

# Corresponding ctypes
RealCType = c_double
IndexCType = c_int64
ByteCType = c_uint8

# Sizes
SIZEOF_REAL = Real.itemsize
SIZEOF_INDEX = Index.itemsize
SIZEOF_BYTE = Byte.itemsize


# =============================================================================
# Alignment Constants
# =============================================================================

# Default alignment for SIMD operations (64 bytes for AVX-512)
SCL_ALIGNMENT = 64

# Cache line size (typical)
CACHE_LINE_SIZE = 64


# =============================================================================
# Type Validation
# =============================================================================

def validate_dtype(dtype: Union[DType, str, Type, None],
                   default: DType = DType.FLOAT64) -> DType:
    """
    Validate and normalize dtype specification.

    Args:
        dtype: Input dtype (DType enum, string name, ctypes type, or None)
        default: Default dtype if None

    Returns:
        Validated DType
    """
    if dtype is None:
        return default
    if isinstance(dtype, DType):
        return dtype
    if isinstance(dtype, str):
        return DType.from_name(dtype)
    if dtype in CTYPE_MAP:
        return CTYPE_MAP[dtype]
    if dtype in TYPE_MAP:
        return TYPE_MAP[dtype]
    raise TypeError(f"Cannot convert {dtype!r} to DType")


def dtype_compatible(a: DType, b: DType) -> bool:
    """Check if two dtypes are compatible (same size and signedness)."""
    if a == b:
        return True
    # Float64 and Int64 are both 8 bytes but not compatible
    # Int32 and Float32 are both 4 bytes but not compatible
    return False


def promote_dtype(a: DType, b: DType) -> DType:
    """
    Promote two dtypes to common type.

    Rules:
    - Float > Int
    - Larger size wins
    - Same category: larger wins
    """
    # Same type
    if a == b:
        return a

    # Float promotion
    if a in (DType.FLOAT32, DType.FLOAT64) or b in (DType.FLOAT32, DType.FLOAT64):
        if a == DType.FLOAT64 or b == DType.FLOAT64:
            return DType.FLOAT64
        return DType.FLOAT32

    # Integer promotion
    if a == DType.INT64 or b == DType.INT64:
        return DType.INT64
    if a == DType.INT32 or b == DType.INT32:
        return DType.INT32

    # Default to larger type
    return a if a.itemsize >= b.itemsize else b


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enum
    "DType",
    # Primary types
    "Real",
    "Index",
    "Byte",
    # CTypes
    "RealCType",
    "IndexCType",
    "ByteCType",
    # Size constants
    "SIZEOF_REAL",
    "SIZEOF_INDEX",
    "SIZEOF_BYTE",
    "SCL_ALIGNMENT",
    "CACHE_LINE_SIZE",
    # Type maps
    "TYPE_MAP",
    "CTYPE_MAP",
    # Utilities
    "validate_dtype",
    "dtype_compatible",
    "promote_dtype",
]
