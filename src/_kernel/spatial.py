"""
Spatial Statistics Kernels

Low-level C bindings for spatial autocorrelation analysis.
"""

import ctypes
import numpy as np
from .lib_loader import get_lib
from .types import c_real, c_index, check_error, as_c_ptr

__all__ = [
    'morans_i',
    'gearys_c',
]

# =============================================================================
# Function Signatures
# =============================================================================

def _init_signatures():
    """Initialize C function signatures."""
    lib = get_lib()
    
    # morans_i
    lib.scl_morans_i.argtypes = [
        ctypes.POINTER(c_real),   # graph_data
        ctypes.POINTER(c_index),  # graph_indices
        ctypes.POINTER(c_index),  # graph_indptr
        ctypes.POINTER(c_index),  # graph_row_lengths
        c_index,                   # n_cells
        c_index,                   # graph_nnz
        ctypes.POINTER(c_real),   # features_data
        ctypes.POINTER(c_index),  # features_indices
        ctypes.POINTER(c_index),  # features_indptr
        ctypes.POINTER(c_index),  # features_col_lengths
        c_index,                   # n_genes
        c_index,                   # features_nnz
        ctypes.POINTER(c_real),   # output
    ]
    lib.scl_morans_i.restype = ctypes.c_int
    
    # gearys_c
    lib.scl_gearys_c.argtypes = [
        ctypes.POINTER(c_real),   # graph_data
        ctypes.POINTER(c_index),  # graph_indices
        ctypes.POINTER(c_index),  # graph_indptr
        ctypes.POINTER(c_index),  # graph_row_lengths
        c_index,                   # n_cells
        c_index,                   # graph_nnz
        ctypes.POINTER(c_real),   # features_data
        ctypes.POINTER(c_index),  # features_indices
        ctypes.POINTER(c_index),  # features_indptr
        ctypes.POINTER(c_index),  # features_col_lengths
        c_index,                   # n_genes
        c_index,                   # features_nnz
        ctypes.POINTER(c_real),   # output
    ]
    lib.scl_gearys_c.restype = ctypes.c_int


_init_signatures()

# =============================================================================
# Python Wrappers
# =============================================================================

def morans_i(
    graph_data: np.ndarray,
    graph_indices: np.ndarray,
    graph_indptr: np.ndarray,
    graph_row_lengths: np.ndarray,
    n_cells: int,
    graph_nnz: int,
    features_data: np.ndarray,
    features_indices: np.ndarray,
    features_indptr: np.ndarray,
    features_col_lengths: np.ndarray,
    n_genes: int,
    features_nnz: int,
    output: np.ndarray
) -> None:
    """
    Compute Moran's I spatial autocorrelation.
    
    Args:
        graph_data: Spatial graph CSR data
        graph_indices: Spatial graph CSR indices
        graph_indptr: Spatial graph CSR indptr
        graph_row_lengths: Explicit lengths or None
        n_cells: Number of cells
        graph_nnz: Graph nnz
        features_data: Feature matrix CSC data
        features_indices: Feature matrix CSC indices
        features_indptr: Feature matrix CSC indptr
        features_col_lengths: Explicit lengths or None
        n_genes: Number of genes
        features_nnz: Features nnz
        output: Output Moran's I values, shape (n_genes,)
    """
    lib = get_lib()
    
    status = lib.scl_morans_i(
        as_c_ptr(graph_data, c_real),
        as_c_ptr(graph_indices, c_index),
        as_c_ptr(graph_indptr, c_index),
        as_c_ptr(graph_row_lengths, c_index) if graph_row_lengths is not None else None,
        n_cells,
        graph_nnz,
        as_c_ptr(features_data, c_real),
        as_c_ptr(features_indices, c_index),
        as_c_ptr(features_indptr, c_index),
        as_c_ptr(features_col_lengths, c_index) if features_col_lengths is not None else None,
        n_genes,
        features_nnz,
        as_c_ptr(output, c_real)
    )
    
    check_error(status, "morans_i")


def gearys_c(
    graph_data: np.ndarray,
    graph_indices: np.ndarray,
    graph_indptr: np.ndarray,
    graph_row_lengths: np.ndarray,
    n_cells: int,
    graph_nnz: int,
    features_data: np.ndarray,
    features_indices: np.ndarray,
    features_indptr: np.ndarray,
    features_col_lengths: np.ndarray,
    n_genes: int,
    features_nnz: int,
    output: np.ndarray
) -> None:
    """
    Compute Geary's C spatial autocorrelation.
    
    Args:
        graph_data: Spatial graph CSR data
        graph_indices: Spatial graph CSR indices
        graph_indptr: Spatial graph CSR indptr
        graph_row_lengths: Explicit lengths or None
        n_cells: Number of cells
        graph_nnz: Graph nnz
        features_data: Feature matrix CSC data
        features_indices: Feature matrix CSC indices
        features_indptr: Feature matrix CSC indptr
        features_col_lengths: Explicit lengths or None
        n_genes: Number of genes
        features_nnz: Features nnz
        output: Output Geary's C values, shape (n_genes,)
    """
    lib = get_lib()
    
    status = lib.scl_gearys_c(
        as_c_ptr(graph_data, c_real),
        as_c_ptr(graph_indices, c_index),
        as_c_ptr(graph_indptr, c_index),
        as_c_ptr(graph_row_lengths, c_index) if graph_row_lengths is not None else None,
        n_cells,
        graph_nnz,
        as_c_ptr(features_data, c_real),
        as_c_ptr(features_indices, c_index),
        as_c_ptr(features_indptr, c_index),
        as_c_ptr(features_col_lengths, c_index) if features_col_lengths is not None else None,
        n_genes,
        features_nnz,
        as_c_ptr(output, c_real)
    )
    
    check_error(status, "gearys_c")

