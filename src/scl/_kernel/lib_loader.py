"""Dynamic library loader for SCL C API.

This module handles platform-specific library loading with lazy initialization.
No external dependencies required.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional


__all__ = ['get_lib', 'LibraryNotFoundError']


class LibraryNotFoundError(Exception):
    """Raised when SCL library cannot be found or loaded."""
    pass


# Global library cache
_lib_cache = {}


def _find_library(precision: str = 'f32') -> Optional[Path]:
    """Search for SCL shared library.
    
    Search order:
        1. Environment variable: SCL_LIBRARY_PATH
        2. Package directory: src/scl/libs/
        3. Build directory: build/
        4. System library paths
    
    Args:
        precision: Library precision ('f32' or 'f64').
        
    Returns:
        Path to library file, or None if not found.
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


def get_lib(precision: Optional[str] = None) -> ctypes.CDLL:
    """Get SCL library handle with lazy initialization.
    
    This function handles:
        - Library discovery
        - Loading and caching
        - Automatic precision detection
    
    Args:
        precision: Force specific precision ('f32' or 'f64'), or None for auto-detect.
        
    Returns:
        ctypes.CDLL library handle.
        
    Raises:
        LibraryNotFoundError: If library cannot be found.
        
    Example:
        >>> lib = get_lib()
        >>> version = lib.scl_version()
    """
    # Auto-detect precision
    if precision is None:
        precision = os.environ.get('SCL_PRECISION', 'f32')
    
    if precision not in ('f32', 'f64'):
        raise ValueError(f"Invalid precision: {precision}. Must be 'f32' or 'f64'")
    
    # Check cache
    if precision in _lib_cache:
        return _lib_cache[precision]
    
    # Find library
    lib_path = _find_library(precision)
    if lib_path is None:
        raise LibraryNotFoundError(
            f"Cannot find SCL {precision} library. "
            f"Please build the library first or set SCL_LIBRARY_PATH."
        )
    
    # Load library
    try:
        lib = ctypes.CDLL(str(lib_path))
        _lib_cache[precision] = lib
        return lib
    except OSError as e:
        raise LibraryNotFoundError(f"Failed to load library from {lib_path}: {e}")

