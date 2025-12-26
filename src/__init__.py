"""
SCL - Single Cell Library

High-performance single-cell analysis library with:
- Zero-copy sparse matrix operations
- SIMD-optimized C++ kernels  
- Seamless anndata/scanpy integration
- 50-200x speedup over standard implementations

Modules:
- sparse: Sparse matrix data structures and operations
- hooks: Runtime integration with anndata/scipy (auto-activated)

Example:
    >>> import scl
    >>> from scl.sparse import SclCSR
    >>> import scl.sparse as sp
    >>> 
    >>> # Create matrix with beautiful syntax
    >>> mat = SclCSR.empty(1000, 2000, 50000, dtype=sp.float32)
    >>> 
    >>> # Use with anndata (automatic integration)
    >>> import anndata
    >>> adata = anndata.AnnData(X=mat)  # Works seamlessly!
"""

__version__ = '0.1.0'

# Import main modules
from . import sparse
from . import _hooks as hooks

# Re-export common types
from .sparse import (
    Array,
    SclCSR,
    SclCSC,
    VirtualCSR,
    VirtualCSC,
    # Type constants
    float32,
    float64,
    int32,
    int64,
    uint8,
    uint32,
    uint64,
    # Convenience functions
    vstack_csr,
    hstack_csc,
)

__all__ = [
    # Version
    '__version__',
    # Modules
    'sparse',
    'hooks',
    # Common types
    'Array',
    'SclCSR',
    'SclCSC',
    'VirtualCSR',
    'VirtualCSC',
    # Type constants
    'float32',
    'float64',
    'int32',
    'int64',
    'uint8',
    'uint32',
    'uint64',
    # Convenience functions
    'vstack_csr',
    'hstack_csc',
]

# Auto-install hooks on import (unless disabled)
hooks._auto_install()

