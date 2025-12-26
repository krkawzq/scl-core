"""
SCL Private Kernel Bindings (_kernel)

This is a private package that provides low-level C-style bindings to SCL kernels.

Architecture:
- Direct ctypes bindings to libscl.so
- Minimal Python wrapper (close to C API)
- Type conversions and error handling
- Used as foundation for high-level public API

Modules:
- lib_loader: Dynamic library loading
- types: C type definitions and conversions
- sparse: Sparse matrix statistics kernels
- qc: Quality control kernels
- normalize: Normalization kernels
- stats: Statistical test kernels (MWU, T-test)
- transform: Data transformation kernels (log1p, softmax)
- algebra: Linear algebra kernels (SpMV, Gram)
- feature: Feature selection kernels (HVG)
- spatial: Spatial statistics kernels

Usage (Internal only):
    from ._kernel import sparse
    
    status = sparse.row_sums_csr(
        data, indices, indptr, row_lengths,
        rows, cols, nnz, output
    )
"""

from . import lib_loader
from . import types
from . import sparse
from . import qc
from . import normalize
from . import stats
from . import transform
from . import algebra
from . import feature
from . import spatial
from . import utils

__all__ = [
    'lib_loader',
    'types',
    'sparse',
    'qc',
    'normalize',
    'stats',
    'transform',
    'algebra',
    'feature',
    'spatial',
    'utils',
]

