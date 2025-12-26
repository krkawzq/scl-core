"""
SCL - Single Cell Library

High-performance single-cell analysis library with:
- Zero-copy sparse matrix operations
- SIMD-optimized C++ kernels  
- Seamless anndata/scanpy integration
- Smart backend management (Custom, Virtual, Mapped)
- Automatic ownership tracking

Modules:
- sparse: Sparse matrix data structures and operations
- hooks: Runtime integration with anndata/scipy (auto-activated)

Architecture:
    ┌──────────────────────────────────────────────┐
    │        SclCSR / SclCSC (Smart Matrix)        │
    ├──────────────────────────────────────────────┤
    │  Backend: CUSTOM | VIRTUAL | MAPPED          │
    │  Ownership: OWNED | BORROWED | VIEW          │
    └──────────────────────────────────────────────┘

Example:
    >>> import scl
    >>> from scl.sparse import SclCSR, Backend
    >>> 
    >>> # Create matrix
    >>> mat = SclCSR.from_dense([[1, 0, 2], [0, 3, 0]])
    >>> 
    >>> # Smart slicing creates Virtual backend
    >>> view = mat[::2, :]
    >>> print(view.backend)  # Backend.VIRTUAL
    >>> 
    >>> # Materialize when needed
    >>> owned = view.to_owned()
    >>> print(owned.backend)  # Backend.CUSTOM
    >>> 
    >>> # Use with anndata (automatic integration)
    >>> import anndata
    >>> adata = anndata.AnnData(X=mat)
"""

__version__ = '0.2.0'

# Import main modules
from . import sparse
from . import _hooks as hooks

# Re-export common types
from .sparse import (
    # Core classes
    Array,
    SclCSR,
    SclCSC,
    CSR,  # Alias
    CSC,  # Alias
    
    # Backend/Ownership enums
    Backend,
    Ownership,
    
    # Type constants
    DType,
    float32,
    float64,
    int32,
    int64,
    uint8,
    
    # Stacking
    vstack_csr,
    hstack_csc,
    vstack,
    hstack,
    
    # Cross-platform conversion
    from_scipy,
    from_anndata,
    to_scipy,
    to_anndata,
    
    # Type checking
    is_sparse_like,
    is_csr_like,
    is_csc_like,
)

__all__ = [
    # Version
    '__version__',
    
    # Modules
    'sparse',
    'hooks',
    
    # Core classes
    'Array',
    'SclCSR',
    'SclCSC',
    'CSR',
    'CSC',
    
    # Backend/Ownership
    'Backend',
    'Ownership',
    
    # Type constants
    'DType',
    'float32',
    'float64',
    'int32',
    'int64',
    'uint8',
    
    # Stacking
    'vstack_csr',
    'hstack_csc',
    'vstack',
    'hstack',
    
    # Cross-platform
    'from_scipy',
    'from_anndata',
    'to_scipy',
    'to_anndata',
    
    # Type checking
    'is_sparse_like',
    'is_csr_like',
    'is_csc_like',
]

# Auto-install hooks on import (unless disabled)
hooks._auto_install()
