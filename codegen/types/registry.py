"""
Type registry for managing C to ctypes mappings.

This module defines the central type registry that maps C types to their
ctypes equivalents, including SCL-specific types like scl_sparse_t.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class TypeKind(Enum):
    """Classification of C types."""
    BUILTIN = auto()      # int, float, double, etc.
    FIXED_WIDTH = auto()  # int32_t, uint64_t, etc.
    SCL_SCALAR = auto()   # scl_real_t, scl_index_t, etc.
    SCL_HANDLE = auto()   # scl_sparse_t, scl_dense_t (opaque pointers)
    STRUCT = auto()       # User-defined structures
    ENUM = auto()         # Enumerations
    TYPEDEF = auto()      # Type aliases
    UNKNOWN = auto()      # Unrecognized types


@dataclass
class TypeInfo:
    """Information about a mapped type."""
    c_name: str
    ctypes_name: str
    kind: TypeKind
    is_pointer: bool = False
    pointee_type: Optional[str] = None
    doc: Optional[str] = None


# =============================================================================
# Built-in C Types
# =============================================================================

BUILTIN_TYPES: dict[str, str] = {
    # Basic integer types
    "int": "c_int",
    "unsigned int": "c_uint",
    "short": "c_short",
    "unsigned short": "c_ushort",
    "long": "c_long",
    "unsigned long": "c_ulong",
    "long long": "c_longlong",
    "unsigned long long": "c_ulonglong",

    # Character types
    "char": "c_char",
    "signed char": "c_char",
    "unsigned char": "c_ubyte",
    "wchar_t": "c_wchar",

    # Floating point
    "float": "c_float",
    "double": "c_double",
    "long double": "c_longdouble",

    # Boolean
    "bool": "c_bool",
    "_Bool": "c_bool",

    # Void
    "void": "None",
}

# =============================================================================
# Fixed-Width Integer Types (stdint.h)
# =============================================================================

FIXED_WIDTH_TYPES: dict[str, str] = {
    # Signed
    "int8_t": "c_int8",
    "int16_t": "c_int16",
    "int32_t": "c_int32",
    "int64_t": "c_int64",

    # Unsigned
    "uint8_t": "c_uint8",
    "uint16_t": "c_uint16",
    "uint32_t": "c_uint32",
    "uint64_t": "c_uint64",

    # Size types
    "size_t": "c_size_t",
    "ssize_t": "c_ssize_t",
    "ptrdiff_t": "c_ssize_t",
    "intptr_t": "c_ssize_t",
    "uintptr_t": "c_size_t",
}

# =============================================================================
# SCL-Specific Types
# =============================================================================

# Scalar types that map to basic ctypes
SCL_SCALAR_TYPES: dict[str, dict[str, str]] = {
    "scl_error_t": {
        "float32": "c_int32",
        "float64": "c_int32",
    },
    "scl_bool_t": {
        "float32": "c_int",
        "float64": "c_int",
    },
    "scl_size_t": {
        "float32": "c_size_t",
        "float64": "c_size_t",
    },
    "scl_real_t": {
        "float32": "c_float",
        "float64": "c_double",
    },
    "scl_index_t": {
        "int16": "c_int16",
        "int32": "c_int32",
        "int64": "c_int64",
    },
}

# Handle types (opaque pointers to structures)
SCL_HANDLE_TYPES: set[str] = {
    "scl_sparse_t",
    "scl_dense_t",
    "scl_sparse_matrix",
    "scl_dense_matrix",
}


class TypeRegistry:
    """
    Central registry for type mappings.

    Handles resolution of C types to their ctypes equivalents,
    including SCL-specific types with configurable precision.
    """

    def __init__(
        self,
        real_type: str = "float64",
        index_type: str = "int64",
    ):
        """
        Initialize the type registry.

        Args:
            real_type: Floating point precision ("float32" or "float64")
            index_type: Integer index type ("int16", "int32", or "int64")
        """
        self.real_type = real_type
        self.index_type = index_type

        # User-defined types (structs, enums, typedefs)
        self._user_types: dict[str, TypeInfo] = {}

    def register_type(self, info: TypeInfo) -> None:
        """Register a user-defined type."""
        self._user_types[info.c_name] = info

    def lookup(self, c_type: str) -> Optional[TypeInfo]:
        """
        Look up type information for a C type name.

        Args:
            c_type: The C type name (e.g., "int", "scl_sparse_t")

        Returns:
            TypeInfo if found, None otherwise
        """
        # Strip any leading/trailing whitespace
        c_type = c_type.strip()

        # Check builtin types
        if c_type in BUILTIN_TYPES:
            return TypeInfo(
                c_name=c_type,
                ctypes_name=BUILTIN_TYPES[c_type],
                kind=TypeKind.BUILTIN,
            )

        # Check fixed-width types
        if c_type in FIXED_WIDTH_TYPES:
            return TypeInfo(
                c_name=c_type,
                ctypes_name=FIXED_WIDTH_TYPES[c_type],
                kind=TypeKind.FIXED_WIDTH,
            )

        # Check SCL scalar types
        if c_type in SCL_SCALAR_TYPES:
            type_map = SCL_SCALAR_TYPES[c_type]
            # Determine which config key to use
            if c_type in ("scl_real_t",):
                ctypes_name = type_map.get(self.real_type, "c_double")
            elif c_type in ("scl_index_t",):
                ctypes_name = type_map.get(self.index_type, "c_int64")
            else:
                # Use first available mapping
                ctypes_name = next(iter(type_map.values()))

            return TypeInfo(
                c_name=c_type,
                ctypes_name=ctypes_name,
                kind=TypeKind.SCL_SCALAR,
            )

        # Check SCL handle types
        if c_type in SCL_HANDLE_TYPES:
            return TypeInfo(
                c_name=c_type,
                ctypes_name="c_void_p",
                kind=TypeKind.SCL_HANDLE,
                doc="Opaque handle type",
            )

        # Check user-defined types
        if c_type in self._user_types:
            return self._user_types[c_type]

        return None

    def get_ctypes_name(self, c_type: str) -> str:
        """
        Get the ctypes name for a C type.

        Args:
            c_type: The C type name

        Returns:
            The ctypes equivalent name, or "c_void_p" for unknown types
        """
        info = self.lookup(c_type)
        if info is not None:
            return info.ctypes_name

        # Unknown type - default to void pointer
        return "c_void_p"

    def is_handle_type(self, c_type: str) -> bool:
        """Check if a type is an opaque handle type."""
        return c_type in SCL_HANDLE_TYPES

    def is_known_type(self, c_type: str) -> bool:
        """Check if a type is recognized."""
        return self.lookup(c_type) is not None
