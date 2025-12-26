"""SCL Sparse Matrix Module.

This module provides high-performance sparse matrix data structures
with automatic backend management, transparent ownership tracking,
and seamless cross-platform interoperability.

Quick Start:
    >>> from scl.sparse import SclCSR, SclCSC, vstack_csr
    >>> 
    >>> # Create from various sources
    >>> mat = SclCSR.from_dense([[1, 0, 2], [0, 3, 0]])
    >>> mat = SclCSR.from_scipy(scipy_csr_matrix)
    >>> mat = from_anndata(adata)
    >>> 
    >>> # Smart slicing (auto backend management)
    >>> view = mat[0:100, :]      # Virtual backend, zero-copy
    >>> subset = mat[[0,5,10], :] # Row selection
    >>> 
    >>> # Stacking operations
    >>> stacked = vstack_csr([mat1, mat2, mat3])
    >>> 
    >>> # Convert to external formats
    >>> scipy_mat = mat.to_scipy()
    >>> adata = to_anndata(mat)

Architecture:
    ┌─────────────────────────────────────────────────┐
    │           SclCSR / SclCSC (Smart Matrix)        │
    ├─────────────────────────────────────────────────┤
    │  Backend: CUSTOM | VIRTUAL | MAPPED             │
    │  Ownership: OWNED | BORROWED | VIEW             │
    ├─────────────────────────────────────────────────┤
    │  Auto: reference chain, slicing strategy,       │
    │        materialization, format conversion       │
    └─────────────────────────────────────────────────┘

Backend Types:
    - CUSTOM: Local arrays with full control
    - VIRTUAL: Zero-copy views for stacking/slicing
    - MAPPED: Memory-mapped files for large data

Ownership Models:
    - OWNED: Matrix owns the data
    - BORROWED: Data from external source (scipy)
    - VIEW: Derived from another matrix

Key Classes:
    - SclCSR: Smart row-oriented sparse matrix
    - SclCSC: Smart column-oriented sparse matrix
    - Array: Lightweight contiguous array
    
Key Functions:
    - vstack_csr, hstack_csc: Stack matrices
    - from_scipy, to_scipy: scipy interop
    - from_anndata, to_anndata: AnnData interop
    - align_rows, align_cols: Matrix alignment
"""

# =============================================================================
# Array (Foundation)
# =============================================================================
from ._array import (
    Array,
    zeros,
    empty,
    ones,
    from_list,
    from_buffer,
)

# =============================================================================
# Data Types
# =============================================================================
from ._dtypes import (
    DType,
    float32,
    float64,
    int32,
    int64,
    uint8,
    validate_dtype,
    normalize_dtype,
)

# =============================================================================
# Backend & Ownership (Internal, but exposed for advanced users)
# =============================================================================
from ._backend import (
    Backend,
    Ownership,
    StorageInfo,
    CustomStorage,
    VirtualStorage,
    MappedStorage,
    ChunkInfo,
)

from ._ownership import (
    RefChain,
    OwnershipTracker,
    ensure_alive,
)

# =============================================================================
# Smart Sparse Matrices (Main API)
# =============================================================================
from ._csr import SclCSR, CSR
from ._csc import SclCSC, CSC

# =============================================================================
# Operations
# =============================================================================
from ._ops import (
    # Stacking
    vstack_csr,
    hstack_csc,
    vstack,
    hstack,
    
    # Format conversion
    convert_format,
    
    # Cross-platform
    from_scipy,
    from_anndata,
    from_numpy,
    to_scipy,
    to_anndata,
    to_numpy,
    
    # Alignment
    align_rows,
    align_cols,
    align_to_categories,
    
    # Statistics
    sum_rows,
    sum_cols,
    mean_rows,
    mean_cols,
    var_rows,
    var_cols,
    
    # Utilities
    concatenate,
    empty_like,
    zeros_like,
)

# =============================================================================
# Type Checking Utilities
# =============================================================================

def is_sparse_like(obj) -> bool:
    """Check if object is a sparse matrix-like type."""
    return isinstance(obj, (SclCSR, SclCSC))


def is_csr_like(obj) -> bool:
    """Check if object is CSR-like."""
    return isinstance(obj, SclCSR)


def is_csc_like(obj) -> bool:
    """Check if object is CSC-like."""
    return isinstance(obj, SclCSC)


# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # ---- Core Classes ----
    'Array',
    'SclCSR',
    'SclCSC',
    'CSR',  # Alias
    'CSC',  # Alias
    
    # ---- Backend/Ownership (advanced) ----
    'Backend',
    'Ownership',
    'StorageInfo',
    'CustomStorage',
    'VirtualStorage', 
    'MappedStorage',
    'ChunkInfo',
    'RefChain',
    'OwnershipTracker',
    'ensure_alive',
    
    # ---- Array Functions ----
    'zeros',
    'empty',
    'ones',
    'from_list',
    'from_buffer',
    
    # ---- Data Types ----
    'DType',
    'float32',
    'float64',
    'int32',
    'int64',
    'uint8',
    'validate_dtype',
    'normalize_dtype',
    
    # ---- Stacking ----
    'vstack_csr',
    'hstack_csc',
    'vstack',
    'hstack',
    
    # ---- Conversion ----
    'convert_format',
    'from_scipy',
    'from_anndata',
    'from_numpy',
    'to_scipy',
    'to_anndata',
    'to_numpy',
    
    # ---- Alignment ----
    'align_rows',
    'align_cols',
    'align_to_categories',
    
    # ---- Statistics ----
    'sum_rows',
    'sum_cols',
    'mean_rows',
    'mean_cols',
    'var_rows',
    'var_cols',
    
    # ---- Utilities ----
    'concatenate',
    'empty_like',
    'zeros_like',
    'is_sparse_like',
    'is_csr_like',
    'is_csc_like',
]


# =============================================================================
# Version
# =============================================================================
__version__ = '0.2.0'
