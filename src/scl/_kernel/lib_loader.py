"""
Dynamic Library Loader for SCL

Handles platform-specific library loading with lazy initialization.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional, Literal
import warnings

__all__ = ['get_lib', 'get_lib_f32', 'get_lib_f64', 'LibraryNotFoundError']


class LibraryNotFoundError(Exception):
    """Raised when SCL library cannot be found or loaded."""
    pass


# Global library cache
_lib_f32 = None
_lib_f64 = None
_initialized = False


def _find_library(precision: str = 'f32') -> Optional[Path]:
    """
    Search for SCL shared library.
    
    Search order:
    1. Environment variable: SCL_LIBRARY_PATH
    2. Package directory: src/scl/libs/
    3. Build directory: build/
    4. System library paths
    
    Args:
        precision: 'f32' or 'f64'
        
    Returns:
        Path to library file, or None if not found
    """
    # Platform-specific library name
    if sys.platform == 'win32':
        lib_name = f'scl_core_{precision}.dll'
    elif sys.platform == 'darwin':
        lib_name = f'libscl_core_{precision}.dylib'
    else:  # Linux
        lib_name = f'libscl_core_{precision}.so'
    
    # Search locations
    search_paths = []
    
    # 1. Environment variable
    if 'SCL_LIBRARY_PATH' in os.environ:
        env_path = Path(os.environ['SCL_LIBRARY_PATH'])
        if env_path.is_dir():
            search_paths.append(env_path)
        else:
            search_paths.append(env_path.parent)
    
    # 2. Package directory
    try:
        package_dir = Path(__file__).parent.parent / 'libs'
        if package_dir.exists():
            search_paths.append(package_dir)
    except:
        pass
    
    # 3. Build directory
    try:
        project_root = Path(__file__).parent.parent.parent.parent
        build_dirs = [
            project_root / 'build' / 'lib',
            project_root / 'build',
            project_root / 'build' / 'Release',
            project_root / 'build' / 'Debug',
        ]
        search_paths.extend([d for d in build_dirs if d.exists()])
    except:
        pass
    
    # Search for library
    for search_path in search_paths:
        lib_path = search_path / lib_name
        if lib_path.exists():
            return lib_path
    
    return None


def get_lib(precision: Optional[Literal['f32', 'f64']] = None) -> ctypes.CDLL:
    """
    Get SCL library handle.
    
    Automatically detects precision from SCL_PRECISION environment variable
    or defaults to f32.
    
    This function handles:
    - Library discovery
    - Loading and caching
    - Lazy function signature initialization
    
    Args:
        precision: Force specific precision ('f32' or 'f64'), or None for auto
        
    Returns:
        ctypes.CDLL library handle
        
    Raises:
        LibraryNotFoundError: If library cannot be found
    
    Example:
        >>> lib = get_lib()
        >>> lib.scl_version()
    """
    global _initialized
    
    # Auto-detect precision
    if precision is None:
        precision = os.environ.get('SCL_PRECISION', 'f32')
    
    if precision == 'f32':
        return get_lib_f32()
    elif precision == 'f64':
        return get_lib_f64()
    else:
        raise ValueError(f"Invalid precision: {precision}. Must be 'f32' or 'f64'")


def get_lib_f32() -> ctypes.CDLL:
    """
    Get float32 precision library.
    
    Returns:
        ctypes.CDLL handle for f32 library
        
    Raises:
        LibraryNotFoundError: If library not found
    """
    global _lib_f32
    
    if _lib_f32 is not None:
        return _lib_f32
    
    # Find library
    lib_path = _find_library('f32')
    if lib_path is None:
        raise LibraryNotFoundError(
            "Cannot find SCL f32 library. "
            "Please build the library first with 'make' or set SCL_LIBRARY_PATH."
        )
    
    # Load library
    try:
        _lib_f32 = ctypes.CDLL(str(lib_path))
    except OSError as e:
        raise LibraryNotFoundError(f"Failed to load library from {lib_path}: {e}")
    
    return _lib_f32


def get_lib_f64() -> ctypes.CDLL:
    """
    Get float64 precision library.
    
    Returns:
        ctypes.CDLL handle for f64 library
        
    Raises:
        LibraryNotFoundError: If library not found
    """
    global _lib_f64
    
    if _lib_f64 is not None:
        return _lib_f64
    
    # Find library
    lib_path = _find_library('f64')
    if lib_path is None:
        raise LibraryNotFoundError(
            "Cannot find SCL f64 library. "
            "Please build the library first with 'make' or set SCL_LIBRARY_PATH."
        )
    
    # Load library
    try:
        _lib_f64 = ctypes.CDLL(str(lib_path))
    except OSError as e:
        raise LibraryNotFoundError(f"Failed to load library from {lib_path}: {e}")
    
    return _lib_f64


def is_library_available(precision: str = 'f32') -> bool:
    """
    Check if library is available without loading it.
    
    Args:
        precision: 'f32' or 'f64'
        
    Returns:
        True if library file exists
    """
    return _find_library(precision) is not None
