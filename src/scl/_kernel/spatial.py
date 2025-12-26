"""Spatial statistics kernels."""
import ctypes
from typing import Any, Optional
from .lib_loader import get_lib
from .types import c_real, c_index, check_error

__all__ = ['morans_i']

def morans_i(graph_data: Any, graph_indices: Any, graph_indptr: Any, graph_row_lengths: Optional[Any],
             n_cells: int, graph_nnz: int, features_data: Any, features_indices: Any, features_indptr: Any,
             features_col_lengths: Optional[Any], n_genes: int, features_nnz: int, output: Any) -> None:
    """Compute Moran's I spatial autocorrelation."""
    lib = get_lib()
    lib.scl_morans_i.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        c_index, c_index,
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        c_index, c_index, ctypes.POINTER(c_real)
    ]
    lib.scl_morans_i.restype = ctypes.c_int
    status = lib.scl_morans_i(graph_data, graph_indices, graph_indptr, graph_row_lengths, n_cells, graph_nnz,
                               features_data, features_indices, features_indptr, features_col_lengths, n_genes, features_nnz, output)
    check_error(status, "morans_i")
