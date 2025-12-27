"""SCL Sparse Matrix Module.

This module provides high-performance sparse matrix data structures
with automatic backend management, transparent ownership tracking,
and seamless cross-platform interoperability.

Type Hierarchy:
    
    SparseBase (ABC)
    ├── CSRBase                       # Row-oriented interface
    │   ├── CustomCSR                 # Contiguous arrays (data, indices, indptr)
    │   ├── VirtualCSR                # Zero-copy views (internal, from slicing)
    │   ├── MappedCustomCSR           # Memory-mapped files
    │   ├── MappedVirtualCSR          # Mapped slices (internal)
    │   ├── SclCSR                    # Smart: auto backend management
    │   └── CallbackCSR               # User-extensible callbacks
    └── CSCBase                       # Column-oriented interface
        ├── CustomCSC, VirtualCSC, MappedCustomCSC, MappedVirtualCSC
        ├── SclCSC                    # Smart: auto backend management
        └── CallbackCSC               # User-extensible callbacks

Quick Start:
    >>> from scl.sparse import SclCSR, CustomCSR, MappedCustomCSR
    >>> 
    >>> # Smart class (recommended for most uses)
    >>> mat = SclCSR.from_scipy(scipy_mat)
    >>> view = mat[::2, :]  # Zero-copy VirtualCSR
    >>> 
    >>> # Direct data structure (for advanced users)
    >>> custom = CustomCSR.from_scipy(scipy_mat)
    >>> 
    >>> # Memory-mapped for large data
    >>> mapped = MappedCustomCSR(
    ...     data_path="data.bin",
    ...     indices_path="indices.bin",
    ...     indptr_path="indptr.bin",
    ...     shape=(1000000, 50000)
    ... )

C++ Equivalents:
    Python Class         -> C++ Type
    ------------------------------------------
    CustomCSR/CSC       -> scl::CustomSparse<T, IsCSR>
    VirtualCSR/CSC      -> scl::VirtualSparse<T, IsCSR>
    MappedCustomCSR/CSC -> scl::io::MappedCustomSparse<T, IsCSR>
    MappedVirtualCSR/CSC-> scl::io::MappedVirtualSparse<T, IsCSR>
    
Backend Types:
    - CUSTOM: Contiguous arrays with direct ownership
    - VIRTUAL: Pointer arrays for zero-copy views
    - MAPPED: Memory-mapped files for out-of-core data

Construction Rules:
    - CustomCSR/CSC: External construction allowed
    - VirtualCSR/CSC: Internal only (from slicing)
    - MappedCustomCSR/CSC: Three bin files required
    - MappedVirtualCSR/CSC: Internal only (from Mapped slicing)
    - SclCSR/SclCSC: Proxy for all types (smart backend)

Key Classes:
    - SclCSR/SclCSC: Smart matrices with auto backend
    - CustomCSR/CSC: Direct data structure access
    - MappedCustomCSR/CSC: Memory-mapped large data
    - CallbackCSR/CSC: User-defined data access
    
Key Functions:
    - vstack_csr, hstack_csc: Stack matrices
    - from_scipy, to_scipy: scipy interop
    - from_anndata, to_anndata: AnnData interop
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
# Base Classes (Abstract Interfaces)
# =============================================================================
from ._base import (
    SparseBase,
    CSRBase,
    CSCBase,
    SparseFormat,
)

# =============================================================================
# Concrete Data Structure Classes (Foundation)
# =============================================================================

# Custom: Contiguous array storage (data, indices, indptr)
# C++ Equivalent: scl::CustomSparse<T, IsCSR>
from ._custom import CustomCSR, CustomCSC

# Virtual: Pointer array storage (zero-copy views, internal use)
# C++ Equivalent: scl::VirtualSparse<T, IsCSR>
from ._virtual import VirtualCSR, VirtualCSC

# Mapped: Memory-mapped file storage (three bin files)
# C++ Equivalent: scl::io::MappedCustomSparse<T, IsCSR>
# MappedVirtual: Zero-copy slicing over mapped storage (internal use)
# C++ Equivalent: scl::io::MappedVirtualSparse<T, IsCSR>
from ._mapped import MappedCustomCSR, MappedCustomCSC, MappedVirtualCSR, MappedVirtualCSC

# =============================================================================
# Smart Sparse Matrices (Main API - Auto Backend Management)
# =============================================================================
from ._csr import SclCSR, CSR
from ._csc import SclCSC, CSC

# =============================================================================
# Callback-Based Sparse Matrices (User-Extensible)
# =============================================================================
from ._callback import CallbackCSR, CallbackCSC

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
    """Check if object is a sparse matrix-like type.
    
    Returns True for:
    - SclCSR, SclCSC (standard matrices)
    - CallbackCSR, CallbackCSC (callback-based matrices)
    - Any subclass of SparseBase
    """
    return isinstance(obj, SparseBase)


def is_csr_like(obj) -> bool:
    """Check if object is CSR-like.
    
    Returns True for:
    - SclCSR (standard CSR)
    - CallbackCSR (callback-based CSR)
    - Any subclass of CSRBase
    """
    return isinstance(obj, CSRBase)


def is_csc_like(obj) -> bool:
    """Check if object is CSC-like.
    
    Returns True for:
    - SclCSC (standard CSC)
    - CallbackCSC (callback-based CSC)
    - Any subclass of CSCBase
    """
    return isinstance(obj, CSCBase)


# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # ---- Base Classes (Abstract) ----
    'SparseBase',
    'CSRBase',
    'CSCBase',
    'SparseFormat',
    
    # ---- Concrete Data Structure Classes ----
    # Custom: Contiguous arrays (data, indices, indptr)
    'CustomCSR',
    'CustomCSC',
    
    # Virtual: Pointer arrays (zero-copy views)
    'VirtualCSR',
    'VirtualCSC',
    
    # Mapped: Memory-mapped files
    'MappedCustomCSR',
    'MappedCustomCSC',
    'MappedVirtualCSR',
    'MappedVirtualCSC',
    
    # ---- Smart Classes (Auto Backend Management) ----
    'Array',
    'SclCSR',
    'SclCSC',
    'CSR',  # Alias for SclCSR
    'CSC',  # Alias for SclCSC
    
    # ---- Callback Classes (User-Extensible) ----
    'CallbackCSR',
    'CallbackCSC',
    
    # ---- Backend/Ownership (Advanced) ----
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
