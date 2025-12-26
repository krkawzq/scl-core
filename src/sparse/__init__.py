"""
SCL Sparse Matrix Module

Provides Python-side sparse matrix data structures with seamless integration
to C++ kernels and scipy.sparse compatibility.

Design Philosophy:
- Keep C++ pure: complex logic and memory management in Python
- Zero-copy where possible: operate on scipy.sparse arrays directly
- No numpy dependency in core: use lightweight Array class
- Type safety: automatic validation and conversion
- Convenience: Pythonic API with sensible defaults

Classes:
- Array: Lightweight contiguous array (no numpy dependency)
- SclCSR: Custom CSR format with row_lengths for optimized operations
- SclCSC: Custom CSC format with col_lengths for optimized operations
- VirtualCSR: Zero-copy wrapper for scipy.sparse.csr_matrix
- VirtualCSC: Zero-copy wrapper for scipy.sparse.csc_matrix

Usage:
    from scl.sparse import Array, SclCSR, VirtualCSR
    
    # Pure Python (no numpy)
    data = Array.from_list([1.0, 2.0, 3.0], dtype='float32')
    
    # From scipy (optional dependency)
    import scipy.sparse as sp
    scipy_mat = sp.csr_matrix(...)
    scl_mat = SclCSR.from_scipy(scipy_mat)
    
    # Zero-copy view
    virtual = VirtualCSR(scipy_mat)
    
    # Back to scipy
    result = scl_mat.to_scipy()
"""

from ._array import Array, empty, zeros, from_list, from_buffer
from ._matrix import SclCSR, SclCSC
from ._virtual_matrix import VirtualCSR, VirtualCSC
from ._dtypes import (
    DType,
    float32, float64,
    int32, int64,
    uint8, uint32, uint64,
    normalize_dtype,
    validate_dtype,
    is_float_dtype,
    is_int_dtype,
    dtype_itemsize,
)

__all__ = [
    # Array
    'Array',
    'empty',
    'zeros',
    'from_list',
    'from_buffer',
    # Matrices
    'SclCSR',
    'SclCSC',
    'VirtualCSR',
    'VirtualCSC',
    # Convenience functions
    'vstack_csr',
    'hstack_csc',
    # Type system
    'DType',
    'float32',
    'float64',
    'int32',
    'int64',
    'uint8',
    'uint32',
    'uint64',
    'normalize_dtype',
    'validate_dtype',
    'is_float_dtype',
    'is_int_dtype',
    'dtype_itemsize',
]


# =============================================================================
# Convenience Functions
# =============================================================================

def vstack_csr(matrices: List[Any]) -> VirtualCSR:
    """
    Vertically stack CSR matrices (zero-copy).
    
    Creates a composite view that logically concatenates matrices.
    No memory is copied until to_owned() is called.
    
    Args:
        matrices: List of CSR matrices (scipy, SclCSR, or VirtualCSR)
        
    Returns:
        VirtualCSR representing the stacked matrix
    
    Example:
        >>> mat1 = sp.csr_matrix(...)  # 1000 × 2000
        >>> mat2 = sp.csr_matrix(...)  # 500 × 2000
        >>> stacked = vstack_csr([mat1, mat2])  # 1500 × 2000, zero-copy
        >>> 
        >>> # Materialize when needed
        >>> owned = stacked.to_owned()
    """
    return VirtualCSR(matrices)


def hstack_csc(matrices: List[Any]) -> VirtualCSC:
    """
    Horizontally stack CSC matrices (zero-copy).
    
    Creates a composite view that logically concatenates matrices.
    
    Args:
        matrices: List of CSC matrices (scipy, SclCSC, or VirtualCSC)
        
    Returns:
        VirtualCSC representing the stacked matrix
    
    Example:
        >>> mat1 = sp.csc_matrix(...)  # 1000 × 2000
        >>> mat2 = sp.csc_matrix(...)  # 1000 × 500
        >>> stacked = hstack_csc([mat1, mat2])  # 1000 × 2500, zero-copy
    """
    return VirtualCSC(matrices)


