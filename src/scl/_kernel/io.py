"""I/O operation kernels.

Low-level C bindings for file I/O, memory-mapped arrays, and HDF5 operations.
"""

import ctypes
from typing import Any, Optional, Tuple

from .lib_loader import get_lib
from .types import c_real, c_index, c_size, c_byte, check_error


__all__ = [
    # Memory-mapped arrays
    'mmap_array_open',
    'mmap_array_prefetch',
    'mmap_array_drop_cache',
    'mmap_array_advise_sequential',
    'mmap_array_advise_random',
    # File utilities
    'file_exists',
    'file_size',
    'create_directory',
    'write_binary_array',
    'read_binary_array',
    'get_file_extension',
    'get_parent_directory',
    'get_filename_stem',
]


# =============================================================================
# Memory-Mapped Arrays
# =============================================================================

def mmap_array_open(
    filepath: str,
    element_size: int,
    writable: bool
) -> Tuple[Any, int]:
    """Open a memory-mapped array from file.
    
    Args:
        filepath: Path to the file.
        element_size: Size of each element in bytes.
        writable: Whether to open for writing.
        
    Returns:
        Tuple of (pointer, size) where size is number of elements.
        
    Raises:
        RuntimeError: If opening fails.
    """
    lib = get_lib()
    lib.scl_mmap_array_open.argtypes = [
        ctypes.c_char_p, c_size, ctypes.c_bool,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(c_size)
    ]
    lib.scl_mmap_array_open.restype = ctypes.c_int
    
    ptr = ctypes.c_void_p()
    size = c_size()
    status = lib.scl_mmap_array_open(
        filepath.encode('utf-8'), element_size, writable,
        ctypes.byref(ptr), ctypes.byref(size)
    )
    check_error(status, "mmap_array_open")
    return (ptr, size.value)


def mmap_array_prefetch(ptr: Any, byte_size: int) -> None:
    """Prefetch memory-mapped data into RAM.
    
    Args:
        ptr: Pointer to mapped memory.
        byte_size: Number of bytes to prefetch.
        
    Raises:
        RuntimeError: If prefetch fails.
    """
    lib = get_lib()
    lib.scl_mmap_array_prefetch.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_mmap_array_prefetch.restype = ctypes.c_int
    
    status = lib.scl_mmap_array_prefetch(ptr, byte_size)
    check_error(status, "mmap_array_prefetch")


def mmap_array_drop_cache(ptr: Any, byte_size: int) -> None:
    """Drop memory-mapped data from cache.
    
    Args:
        ptr: Pointer to mapped memory.
        byte_size: Number of bytes to drop.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_mmap_array_drop_cache.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_mmap_array_drop_cache.restype = ctypes.c_int
    
    status = lib.scl_mmap_array_drop_cache(ptr, byte_size)
    check_error(status, "mmap_array_drop_cache")


def mmap_array_advise_sequential(ptr: Any, byte_size: int) -> None:
    """Advise kernel that access will be sequential.
    
    Args:
        ptr: Pointer to mapped memory.
        byte_size: Number of bytes.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_mmap_array_advise_sequential.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_mmap_array_advise_sequential.restype = ctypes.c_int
    
    status = lib.scl_mmap_array_advise_sequential(ptr, byte_size)
    check_error(status, "mmap_array_advise_sequential")


def mmap_array_advise_random(ptr: Any, byte_size: int) -> None:
    """Advise kernel that access will be random.
    
    Args:
        ptr: Pointer to mapped memory.
        byte_size: Number of bytes.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_mmap_array_advise_random.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_mmap_array_advise_random.restype = ctypes.c_int
    
    status = lib.scl_mmap_array_advise_random(ptr, byte_size)
    check_error(status, "mmap_array_advise_random")


# =============================================================================
# File Utilities
# =============================================================================

def file_exists(filepath: str) -> bool:
    """Check if a file exists.
    
    Args:
        filepath: Path to check.
        
    Returns:
        True if file exists, False otherwise.
        
    Raises:
        RuntimeError: If check fails.
    """
    lib = get_lib()
    lib.scl_file_exists.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.scl_file_exists.restype = ctypes.c_int
    
    exists = ctypes.c_int()
    status = lib.scl_file_exists(filepath.encode('utf-8'), ctypes.byref(exists))
    check_error(status, "file_exists")
    return bool(exists.value)


def file_size(filepath: str) -> int:
    """Get file size in bytes.
    
    Args:
        filepath: Path to file.
        
    Returns:
        File size in bytes.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_file_size.argtypes = [ctypes.c_char_p, ctypes.POINTER(c_size)]
    lib.scl_file_size.restype = ctypes.c_int
    
    size = c_size()
    status = lib.scl_file_size(filepath.encode('utf-8'), ctypes.byref(size))
    check_error(status, "file_size")
    return size.value


def create_directory(dirpath: str) -> None:
    """Create a directory (including parents).
    
    Args:
        dirpath: Path to directory.
        
    Raises:
        RuntimeError: If creation fails.
    """
    lib = get_lib()
    lib.scl_create_directory.argtypes = [ctypes.c_char_p]
    lib.scl_create_directory.restype = ctypes.c_int
    
    status = lib.scl_create_directory(dirpath.encode('utf-8'))
    check_error(status, "create_directory")


def write_binary_array(
    filepath: str,
    data: Any,
    element_size: int,
    num_elements: int
) -> None:
    """Write array to binary file.
    
    Args:
        filepath: Output file path.
        data: Data pointer.
        element_size: Size of each element in bytes.
        num_elements: Number of elements.
        
    Raises:
        RuntimeError: If write fails.
    """
    lib = get_lib()
    lib.scl_write_binary_array.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, c_size, c_size
    ]
    lib.scl_write_binary_array.restype = ctypes.c_int
    
    status = lib.scl_write_binary_array(
        filepath.encode('utf-8'), data, element_size, num_elements
    )
    check_error(status, "write_binary_array")


def read_binary_array(
    filepath: str,
    data: Any,
    element_size: int,
    num_elements: int
) -> None:
    """Read array from binary file.
    
    Args:
        filepath: Input file path.
        data: Data pointer (modified in-place).
        element_size: Size of each element in bytes.
        num_elements: Number of elements.
        
    Raises:
        RuntimeError: If read fails.
    """
    lib = get_lib()
    lib.scl_read_binary_array.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, c_size, c_size
    ]
    lib.scl_read_binary_array.restype = ctypes.c_int
    
    status = lib.scl_read_binary_array(
        filepath.encode('utf-8'), data, element_size, num_elements
    )
    check_error(status, "read_binary_array")


def get_file_extension(filepath: str) -> str:
    """Get file extension.
    
    Args:
        filepath: File path.
        
    Returns:
        File extension (without dot).
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_get_file_extension.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.scl_get_file_extension.restype = ctypes.c_int
    
    ext_buffer = ctypes.create_string_buffer(256)
    status = lib.scl_get_file_extension(filepath.encode('utf-8'), ext_buffer)
    check_error(status, "get_file_extension")
    return ext_buffer.value.decode('utf-8')


def get_parent_directory(filepath: str) -> str:
    """Get parent directory of file.
    
    Args:
        filepath: File path.
        
    Returns:
        Parent directory path.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_get_parent_directory.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.scl_get_parent_directory.restype = ctypes.c_int
    
    dir_buffer = ctypes.create_string_buffer(1024)
    status = lib.scl_get_parent_directory(filepath.encode('utf-8'), dir_buffer)
    check_error(status, "get_parent_directory")
    return dir_buffer.value.decode('utf-8')


def get_filename_stem(filepath: str) -> str:
    """Get filename without extension.
    
    Args:
        filepath: File path.
        
    Returns:
        Filename stem (without extension).
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_get_filename_stem.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.scl_get_filename_stem.restype = ctypes.c_int
    
    name_buffer = ctypes.create_string_buffer(512)
    status = lib.scl_get_filename_stem(filepath.encode('utf-8'), name_buffer)
    check_error(status, "get_filename_stem")
    return name_buffer.value.decode('utf-8')

