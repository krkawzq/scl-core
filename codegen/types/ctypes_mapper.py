"""
C type to ctypes expression mapper.

Handles complex type expressions including pointers, arrays,
and nested type constructs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .registry import TypeRegistry, SCL_HANDLE_TYPES


@dataclass
class CTypeExpr:
    """
    Parsed C type expression.

    Represents a C type including qualifiers and pointer levels.
    """
    base_type: str
    pointer_depth: int = 0
    is_const: bool = False
    array_size: Optional[int] = None

    @property
    def is_pointer(self) -> bool:
        return self.pointer_depth > 0

    @property
    def is_array(self) -> bool:
        return self.array_size is not None


class CtypesMapper:
    """
    Maps C type expressions to ctypes code strings.

    Handles:
    - Basic types (int, float, etc.)
    - Pointer types (int*, const char**)
    - SCL handle types (scl_sparse_t -> POINTER(c_void_p))
    - Struct pointers (struct foo* -> POINTER(FooStruct))
    """

    def __init__(self, registry: TypeRegistry):
        """
        Initialize the mapper.

        Args:
            registry: Type registry for base type lookups
        """
        self.registry = registry

    def parse_type(self, type_str: str) -> CTypeExpr:
        """
        Parse a C type string into a CTypeExpr.

        Args:
            type_str: C type string (e.g., "const int*", "scl_sparse_t**")

        Returns:
            Parsed CTypeExpr
        """
        type_str = type_str.strip()

        # Handle const qualifier
        is_const = False
        if type_str.startswith("const "):
            is_const = True
            type_str = type_str[6:].strip()

        # Count and remove pointer stars
        pointer_depth = 0
        while type_str.endswith("*"):
            pointer_depth += 1
            type_str = type_str[:-1].strip()

        # Handle "const" after base type (e.g., "char const *")
        if type_str.endswith(" const"):
            is_const = True
            type_str = type_str[:-6].strip()

        # Handle struct prefix
        if type_str.startswith("struct "):
            type_str = type_str[7:].strip()

        # Handle array notation [N]
        array_size = None
        if "[" in type_str:
            base, bracket = type_str.split("[", 1)
            type_str = base.strip()
            size_str = bracket.rstrip("]").strip()
            if size_str:
                try:
                    array_size = int(size_str)
                except ValueError:
                    pass

        return CTypeExpr(
            base_type=type_str,
            pointer_depth=pointer_depth,
            is_const=is_const,
            array_size=array_size,
        )

    def map_type(self, type_str: str) -> str:
        """
        Map a C type string to a ctypes expression.

        Args:
            type_str: C type string

        Returns:
            ctypes expression string (e.g., "POINTER(c_int)")
        """
        expr = self.parse_type(type_str)
        return self._map_expr(expr)

    def _map_expr(self, expr: CTypeExpr) -> str:
        """Map a parsed type expression to ctypes."""
        base_type = expr.base_type

        # Special handling for void pointers
        if base_type == "void":
            if expr.pointer_depth == 0:
                return "None"
            elif expr.pointer_depth == 1:
                return "c_void_p"
            else:
                # void** -> POINTER(c_void_p)
                result = "c_void_p"
                for _ in range(expr.pointer_depth - 1):
                    result = f"POINTER({result})"
                return result

        # Special handling for char pointers (strings)
        if base_type == "char" and expr.pointer_depth >= 1:
            if expr.is_const:
                return "c_char_p"
            else:
                # char* (mutable) -> POINTER(c_char)
                result = "c_char"
                for _ in range(expr.pointer_depth):
                    result = f"POINTER({result})"
                return result

        # Get base ctypes name
        ctypes_base = self.registry.get_ctypes_name(base_type)

        # Handle SCL handle types specially
        # scl_sparse_t is typedef scl_sparse_matrix*, so it's already a pointer
        # Therefore:
        #   scl_sparse_t   -> c_void_p           (already a pointer)
        #   scl_sparse_t*  -> POINTER(c_void_p)  (pointer to pointer)
        #   scl_sparse_t** -> POINTER(POINTER(c_void_p))
        if base_type in SCL_HANDLE_TYPES:
            if base_type.endswith("_t"):
                # scl_sparse_t, scl_dense_t - these are pointer typedefs
                # The handle itself is c_void_p
                if expr.pointer_depth == 0:
                    return "c_void_p"
                else:
                    # pointer_depth >= 1
                    result = "c_void_p"
                    for _ in range(expr.pointer_depth):
                        result = f"POINTER({result})"
                    return result
            else:
                # scl_sparse_matrix, scl_dense_matrix - raw struct types
                # A pointer to struct is c_void_p (opaque)
                if expr.pointer_depth == 0:
                    # Raw struct - shouldn't happen in FFI, but handle it
                    return "c_void_p"
                elif expr.pointer_depth == 1:
                    # struct* -> c_void_p
                    return "c_void_p"
                else:
                    # struct** -> POINTER(c_void_p), etc.
                    result = "c_void_p"
                    for _ in range(expr.pointer_depth - 1):
                        result = f"POINTER({result})"
                    return result

        # Apply pointer depth
        result = ctypes_base
        for _ in range(expr.pointer_depth):
            result = f"POINTER({result})"

        # Handle arrays
        if expr.array_size is not None:
            result = f"{ctypes_base} * {expr.array_size}"

        return result

    def map_function_args(self, param_types: list[str]) -> list[str]:
        """
        Map function parameter types to ctypes argtypes list.

        Args:
            param_types: List of C type strings

        Returns:
            List of ctypes expression strings
        """
        return [self.map_type(t) for t in param_types]

    def map_return_type(self, return_type: str) -> str:
        """
        Map a function return type to ctypes restype.

        Args:
            return_type: C return type string

        Returns:
            ctypes expression string
        """
        return self.map_type(return_type)
