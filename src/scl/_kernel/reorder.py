"""Reordering kernels.

Low-level C bindings for matrix reordering operations.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_index, check_error


__all__ = ['align_secondary_csc']


def align_secondary_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    index_map: Any,
    out_lengths: Any,
    new_cols: int
) -> None:
    """Align secondary dimension indices for CSC matrix.
    
    Reorders row indices according to a mapping, used when aligning
    cells across different matrices (e.g., in AnnData concatenation).
    
    Args:
        data: CSC data array pointer (modified in-place).
        indices: CSC row indices pointer (modified in-place).
        indptr: CSC column pointers pointer.
        rows: Original number of rows.
        cols: Number of columns.
        index_map: Mapping from old row indices to new [rows].
        out_lengths: Output column lengths pointer [cols].
        new_cols: New number of columns after alignment.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_align_secondary_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, ctypes.c_void_p, ctypes.c_void_p, c_index
    ]
    lib.scl_align_secondary_csc.restype = ctypes.c_int
    
    status = lib.scl_align_secondary_csc(
        data, indices, indptr, rows, cols,
        index_map, out_lengths, new_cols
    )
    check_error(status, "align_secondary_csc")
