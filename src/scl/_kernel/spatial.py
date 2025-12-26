"""Spatial statistics kernels.

Low-level C bindings for spatial autocorrelation statistics.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_real, c_index, check_error


__all__ = [
    'morans_i',
]


def morans_i(
    graph_data: Any,
    graph_indices: Any,
    graph_indptr: Any,
    n_cells: int,
    features_data: Any,
    features_indices: Any,
    features_indptr: Any,
    n_genes: int,
    output: Any
) -> None:
    """Compute Moran's I spatial autocorrelation statistic.
    
    Moran's I measures spatial autocorrelation of features based on a
    spatial connectivity graph. Values range from -1 (dispersed) to +1
    (clustered), with 0 indicating random spatial distribution.
    
    Args:
        graph_data: Spatial graph weights (CSR) data pointer.
        graph_indices: Spatial graph (CSR) column indices pointer.
        graph_indptr: Spatial graph (CSR) row pointers pointer.
        n_cells: Number of cells (rows in graph).
        features_data: Feature matrix (CSC) data pointer.
        features_indices: Feature matrix (CSC) row indices pointer.
        features_indptr: Feature matrix (CSC) column pointers pointer.
        n_genes: Number of genes (columns in features).
        output: Output Moran's I values [n_genes].
        
    Raises:
        RuntimeError: If C function fails.
        
    Notes:
        - Graph should be row-normalized spatial connectivity weights
        - Features should be centered for proper interpretation
        - Positive I indicates spatial clustering
        - Negative I indicates spatial dispersion
    """
    lib = get_lib()
    lib.scl_morans_i.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_index,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_index,
        ctypes.c_void_p
    ]
    lib.scl_morans_i.restype = ctypes.c_int
    
    status = lib.scl_morans_i(
        graph_data, graph_indices, graph_indptr, n_cells,
        features_data, features_indices, features_indptr, n_genes,
        output
    )
    check_error(status, "morans_i")
