"""
Type system for C to Python ctypes mapping.
"""

from .registry import TypeRegistry
from .ctypes_mapper import CtypesMapper

__all__ = ["TypeRegistry", "CtypesMapper"]
