"""
Global configuration for SCL Python interface.

Provides:
- Default precision settings (real type, index type)
- Lazy library loading
- Automatic library dispatch based on data types
"""

from __future__ import annotations

import ctypes
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple, Union
from functools import lru_cache

import numpy as np

if TYPE_CHECKING:
    from .._bindings._loader import Library


# =============================================================================
# Precision Types
# =============================================================================

class RealType(Enum):
    """Real (floating-point) precision."""
    FLOAT32 = "f32"
    FLOAT64 = "f64"
    
    @property
    def numpy_dtype(self) -> np.dtype:
        return np.float32 if self == RealType.FLOAT32 else np.float64
    
    @property
    def ctypes_type(self):
        return ctypes.c_float if self == RealType.FLOAT32 else ctypes.c_double


class IndexType(Enum):
    """Index (integer) precision."""
    INT32 = "i32"
    INT64 = "i64"
    
    @property
    def numpy_dtype(self) -> np.dtype:
        return np.int32 if self == IndexType.INT32 else np.int64
    
    @property
    def ctypes_type(self):
        return ctypes.c_int32 if self == IndexType.INT32 else ctypes.c_int64


# =============================================================================
# Global Configuration State
# =============================================================================

class _Config:
    """
    Global configuration singleton.
    
    Manages default precision and lazy library loading.
    """
    
    def __init__(self):
        # Default: float64 + int64 (most compatible)
        self._default_real = RealType.FLOAT64
        self._default_index = IndexType.INT64
        
        # Cached library instances (lazy loaded)
        self._libraries: dict[str, "Library"] = {}
    
    @property
    def default_real(self) -> RealType:
        """Get default real type."""
        return self._default_real
    
    @default_real.setter
    def default_real(self, value: Union[RealType, str]):
        """Set default real type."""
        if isinstance(value, str):
            value = RealType(value) if value in ('f32', 'f64') else \
                    RealType.FLOAT32 if 'float32' in value.lower() or '32' in value else \
                    RealType.FLOAT64
        self._default_real = value
    
    @property
    def default_index(self) -> IndexType:
        """Get default index type."""
        return self._default_index
    
    @default_index.setter
    def default_index(self, value: Union[IndexType, str]):
        """Set default index type."""
        if isinstance(value, str):
            value = IndexType(value) if value in ('i32', 'i64') else \
                    IndexType.INT32 if 'int32' in value.lower() or '32' in value else \
                    IndexType.INT64
        self._default_index = value
    
    @property
    def default_variant(self) -> str:
        """Get default library variant name."""
        return f"{self._default_real.value}_{self._default_index.value}"
    
    def get_library(self, variant: Optional[str] = None) -> "Library":
        """
        Get library instance (lazy loaded).
        
        Args:
            variant: Library variant (e.g., "f64_i64"). If None, uses default.
            
        Returns:
            Library instance
        """
        if variant is None:
            variant = self.default_variant
        
        if variant not in self._libraries:
            self._libraries[variant] = self._load_library(variant)
        
        return self._libraries[variant]
    
    def _load_library(self, variant: str) -> "Library":
        """Load library variant."""
        from .._bindings._loader import Library
        return Library(variant)
    
    def infer_variant_from_data(
        self,
        data: Optional[np.ndarray] = None,
        index: Optional[np.ndarray] = None,
    ) -> str:
        """
        Infer library variant from data types.
        
        Args:
            data: Data array (for real type inference)
            index: Index array (for index type inference)
            
        Returns:
            Library variant string (e.g., "f64_i64")
            
        Note:
            Currently always uses int64 indices for maximum compatibility
            and to support large matrices. This may be configurable in the future.
        """
        # Infer real type
        if data is not None:
            if data.dtype in (np.float32,):
                real = RealType.FLOAT32
            else:
                real = RealType.FLOAT64
        else:
            real = self._default_real
        
        # Always use int64 for indices (safer, supports larger matrices)
        # TODO: Re-enable int32 optimization once C++ side is fixed
        idx = IndexType.INT64
        
        return f"{real.value}_{idx.value}"
    
    def get_library_for_data(
        self,
        data: Optional[np.ndarray] = None,
        index: Optional[np.ndarray] = None,
    ) -> "Library":
        """
        Get appropriate library for data types.
        
        Automatically selects library variant based on array dtypes.
        """
        variant = self.infer_variant_from_data(data, index)
        return self.get_library(variant)


# Global config instance
_config = _Config()


# =============================================================================
# Public API
# =============================================================================

def get_config() -> _Config:
    """Get global configuration instance."""
    return _config


def set_precision(
    real: Optional[Union[RealType, str]] = None,
    index: Optional[Union[IndexType, str]] = None,
) -> None:
    """
    Set default precision for SCL operations.
    
    Args:
        real: Real type ('float32', 'float64', 'f32', 'f64')
        index: Index type ('int32', 'int64', 'i32', 'i64')
    
    Example:
        >>> scl.set_precision(real='float32', index='int32')
        >>> mat = SparseMatrix.copy(scipy_mat)  # Uses f32_i32
    """
    if real is not None:
        _config.default_real = real
    if index is not None:
        _config.default_index = index


def get_precision() -> Tuple[RealType, IndexType]:
    """
    Get current default precision.
    
    Returns:
        Tuple of (real_type, index_type)
    """
    return (_config.default_real, _config.default_index)


def get_library(variant: Optional[str] = None) -> "Library":
    """
    Get library instance (lazy loaded).
    
    Args:
        variant: Specific variant (e.g., "f64_i64"). If None, uses default.
        
    Returns:
        Library instance
    """
    return _config.get_library(variant)


def get_default_library() -> "Library":
    """
    Get default library based on current precision settings.
    
    This is lazy-loaded on first call.
    """
    return _config.get_library()


# =============================================================================
# Internal Helpers (for use by SparseMatrix, etc.)
# =============================================================================

def _infer_library(
    data: Optional[np.ndarray] = None,
    indices: Optional[np.ndarray] = None,
) -> "Library":
    """
    Internal: Infer and get appropriate library for data.
    
    Used by SparseMatrix.copy(), etc. to auto-select library.
    """
    return _config.get_library_for_data(data, indices)


def _get_real_dtype() -> np.dtype:
    """Get NumPy dtype for current default real type."""
    return _config.default_real.numpy_dtype


def _get_index_dtype() -> np.dtype:
    """Get NumPy dtype for current default index type."""
    return _config.default_index.numpy_dtype

