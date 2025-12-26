"""
Dynamic Library Loader for SCL

Handles platform-specific library loading and symbol resolution.
Supports both f32 and f64 precision versions.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional, Literal

__all__ = ['get_lib', 'get_lib_f32', 'get_lib_f64', 'LibraryNotFoundError']


class LibraryNotFoundError(Exception):
    """Raised when SCL library cannot be found or loaded."""
    pass


def _get_library_name(base_name: str) -> str:
    """Get platform-specific library name."""
    if sys.platform == 'win32':
        return f"{base_name}.dll"
    elif sys.platform == 'darwin':
        return f"lib{base_name}.dylib"
    else:
        return f"lib{base_name}.so"


def _find_library(precision: Literal['f32', 'f64'] = 'f32') -> Optional[Path]:
    """
    Search for SCL shared library in standard locations.
    
    Search order:
    1. Environment variable: SCL_LIBRARY_PATH
    2. Build directory: build/lib/
    3. Package libs directory: src/scl_core/libs/
    4. System library paths
    
    Args:
        precision: Library precision version ('f32' or 'f64')
    
    Returns:
        Path to library file, or None if not found
    """
    base_name = f"scl_core_{precision}"
    lib_name = _get_library_name(base_name)
    
    # 1. Check environment variable
    env_path = os.environ.get('SCL_LIBRARY_PATH')
    if env_path:
        lib_path = Path(env_path) / lib_name
        if lib_path.exists():
            return lib_path.resolve()
    
    # 2. Check build directory (build/lib/)
    pkg_dir = Path(__file__).parent
    build_dir = pkg_dir / '..' / '..' / '..' / 'build' / 'lib'
    lib_path = build_dir / lib_name
    if lib_path.exists():
        return lib_path.resolve()
    
    # Also check build root (for backward compatibility)
    build_root = pkg_dir / '..' / '..' / '..' / 'build'
    lib_path = build_root / lib_name
    if lib_path.exists():
        return lib_path.resolve()
    
    # 3. Check installed package libs directory (scl/libs/)
    # This is the standard location after installation
    libs_dir = pkg_dir / '..' / 'libs'
    lib_path = libs_dir / lib_name
    if lib_path.exists():
        return lib_path.resolve()
    
    # 4. Check alternative libs directory (for development)
    alt_libs_dir = pkg_dir / '..' / '..' / '..' / 'libs'
    lib_path = alt_libs_dir / lib_name
    if lib_path.exists():
        return lib_path.resolve()
    
    return None


# Global library handles (singleton per precision)
_lib_f32: Optional[ctypes.CDLL] = None
_lib_f64: Optional[ctypes.CDLL] = None


def get_lib(precision: Optional[Literal['f32', 'f64']] = None) -> ctypes.CDLL:
    """
    Get the loaded SCL library handle (singleton pattern).
    
    Args:
        precision: Library precision version ('f32' or 'f64').
                   If None, defaults to 'f32' or from SCL_PRECISION env var.
    
    Returns:
        ctypes.CDLL handle to libscl
        
    Raises:
        LibraryNotFoundError: If library cannot be found or loaded
    """
    # Determine precision
    if precision is None:
        precision = os.environ.get('SCL_PRECISION', 'f32')
    if precision not in ('f32', 'f64'):
        raise ValueError(f"precision must be 'f32' or 'f64', got {precision}")
    
    # Get the appropriate library handle
    if precision == 'f32':
        return get_lib_f32()
    else:
        return get_lib_f64()


def get_lib_f32() -> ctypes.CDLL:
    """
    Get the loaded SCL f32 library handle.
    
    Returns:
        ctypes.CDLL handle to scl_core_f32
    """
    global _lib_f32
    
    if _lib_f32 is not None:
        return _lib_f32
    
    lib_path = _find_library('f32')
    if lib_path is None:
        raise LibraryNotFoundError(
            "Cannot find SCL f32 library. "
            "Please build the library first or set SCL_LIBRARY_PATH."
        )
    
    try:
        _lib_f32 = ctypes.CDLL(str(lib_path))
    except OSError as e:
        raise LibraryNotFoundError(
            f"Failed to load SCL f32 library from {lib_path}: {e}"
        )
    
    return _lib_f32


def get_lib_f64() -> ctypes.CDLL:
    """
    Get the loaded SCL f64 library handle.
    
    Returns:
        ctypes.CDLL handle to scl_core_f64
    """
    global _lib_f64
    
    if _lib_f64 is not None:
        return _lib_f64
    
    lib_path = _find_library('f64')
    if lib_path is None:
        raise LibraryNotFoundError(
            "Cannot find SCL f64 library. "
            "Please build the library first or set SCL_LIBRARY_PATH."
        )
    
    try:
        _lib_f64 = ctypes.CDLL(str(lib_path))
    except OSError as e:
        raise LibraryNotFoundError(
            f"Failed to load SCL f64 library from {lib_path}: {e}"
        )
    
    return _lib_f64


def check_version() -> str:
    """
    Get SCL library version.
    
    Returns:
        Version string (e.g., "0.1.0")
    """
    lib = get_lib()
    lib.scl_version.restype = ctypes.c_char_p
    version = lib.scl_version()
    return version.decode('utf-8')


def check_precision() -> str:
    """
    Get SCL precision type.
    
    Returns:
        Precision name: "float32" or "float64"
    """
    lib = get_lib()
    lib.scl_precision_name.restype = ctypes.c_char_p
    name = lib.scl_precision_name()
    return name.decode('utf-8')

