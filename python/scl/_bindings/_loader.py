"""
Native Library Loader for SCL

Handles loading the correct shared library based on platform and precision.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import Optional

_lib_cache: dict[str, ctypes.CDLL] = {}

def _get_lib_path() -> Path:
    """Get the path to the native library directory."""
    # Check environment variable first
    if "SCL_LIB_PATH" in os.environ:
        return Path(os.environ["SCL_LIB_PATH"])
    
    # Default: relative to this file
    return Path(__file__).parent.parent / "libs"


def _get_lib_name(precision: str = "f32_i32") -> str:
    """Get the library filename for current platform."""
    system = platform.system()
    
    base_name = f"scl_{precision}"
    
    if system == "Windows":
        return f"{base_name}.dll"
    elif system == "Darwin":
        return f"lib{base_name}.dylib"
    else:  # Linux and others
        return f"lib{base_name}.so"


def get_library(precision: str = "f32_i32") -> ctypes.CDLL:
    """
    Load and return the native SCL library.
    
    Args:
        precision: Library variant to load ("f32_i32" or "f64_i64")
    
    Returns:
        Loaded ctypes.CDLL instance
    
    Raises:
        OSError: If library cannot be found or loaded
    """
    if precision in _lib_cache:
        return _lib_cache[precision]
    
    lib_dir = _get_lib_path()
    lib_name = _get_lib_name(precision)
    lib_path = lib_dir / lib_name
    
    if not lib_path.exists():
        # Try finding in system paths
        try:
            lib = ctypes.CDLL(lib_name)
            _lib_cache[precision] = lib
            return lib
        except OSError:
            pass
        
        raise OSError(
            f"Cannot find SCL library: {lib_path}\n"
            f"Searched in: {lib_dir}\n"
            f"Set SCL_LIB_PATH environment variable to specify custom path."
        )
    
    lib = ctypes.CDLL(str(lib_path))
    _lib_cache[precision] = lib
    return lib


def get_available_precisions() -> list[str]:
    """Get list of available library precisions."""
    lib_dir = _get_lib_path()
    precisions = []
    
    for variant in ["f32_i32", "f64_i64", "f32_i64", "f64_i32"]:
        lib_name = _get_lib_name(variant)
        if (lib_dir / lib_name).exists():
            precisions.append(variant)
    
    return precisions


# Convenience: default library instance
_default_lib: Optional[ctypes.CDLL] = None

def get_default_library() -> ctypes.CDLL:
    """Get the default library (f32_i32 precision)."""
    global _default_lib
    if _default_lib is None:
        _default_lib = get_library("f32_i32")
    return _default_lib
