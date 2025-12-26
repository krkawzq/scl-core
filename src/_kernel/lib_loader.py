"""
Dynamic Library Loader for SCL

Handles platform-specific library loading and symbol resolution.
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


def _find_library() -> Optional[Path]:
    """
    Search for libscl shared library in standard locations.
    
    Search order:
    1. Environment variable: SCL_LIBRARY_PATH
    2. Package directory: src/_kernel/../../../build/
    3. System library paths
    
    Returns:
        Path to library file, or None if not found
    """
    # Platform-specific library name
    if sys.platform == 'win32':
        lib_name = 'scl.dll'
    elif sys.platform == 'darwin':
        lib_name = 'libscl.dylib'
    else:
        lib_name = 'libscl.so'
    
    # 1. Check environment variable
    env_path = os.environ.get('SCL_LIBRARY_PATH')
    if env_path:
        lib_path = Path(env_path) / lib_name
        if lib_path.exists():
            return lib_path
    
    # 2. Check build directory (relative to this file)
    pkg_dir = Path(__file__).parent
    build_dir = pkg_dir / '..' / '..' / '..' / 'build'
    lib_path = build_dir / lib_name
    if lib_path.exists():
        return lib_path.resolve()
    
    # 3. Check libs directory
    libs_dir = pkg_dir / '..' / '..' / '..' / 'libs'
    lib_path = libs_dir / lib_name
    if lib_path.exists():
        return lib_path.resolve()
    
    return None


# Global library handle (singleton)
_lib: Optional[ctypes.CDLL] = None


def get_lib() -> ctypes.CDLL:
    """
    Get the loaded SCL library handle (singleton pattern).
    
    Returns:
        ctypes.CDLL handle to libscl
        
    Raises:
        LibraryNotFoundError: If library cannot be found or loaded
    """
    global _lib
    
    if _lib is not None:
        return _lib
    
    # Find library
    lib_path = _find_library()
    
    if lib_path is None:
        raise LibraryNotFoundError(
            "Cannot find SCL library. "
            "Please set SCL_LIBRARY_PATH environment variable "
            "or build the library first."
        )
    
    # Load library
    try:
        _lib = ctypes.CDLL(str(lib_path))
    except OSError as e:
        raise LibraryNotFoundError(
            f"Failed to load SCL library from {lib_path}: {e}"
        )
    
    return _lib


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

