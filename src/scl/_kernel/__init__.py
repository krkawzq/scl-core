"""SCL Private Kernel Bindings (_kernel).

This is a private package that provides low-level C-style bindings to SCL kernels.

Architecture:
    - Direct ctypes bindings to libscl.so
    - Minimal Python wrapper (close to C API)
    - Type conversions and error handling
    - Used as foundation for high-level public API

Design Principles:
    - No external dependencies (numpy/scipy imported lazily only when needed)
    - Pure C API wrappers without high-level logic
    - Google-style docstrings with type hints
    - One module per C API section

Modules:
    - lib_loader: Dynamic library loading
    - types: C type definitions and error handling
    - sparse: Sparse matrix statistics (Section 3)
    - qc: Quality control metrics (Section 4)
    - normalize: Normalization operations (Section 5)
    - feature: Feature statistics (Section 6)
    - stats: Statistical tests (Section 7)
    - transform: Data transformations (Section 8)
    - algebra: Linear algebra operations (Sections 9, 10, 16)
    - group: Group aggregations (Section 11)
    - scale: Standardization (Section 12)
    - mmd: Maximum Mean Discrepancy (Section 14)
    - spatial: Spatial statistics (Section 15)
    - hvg: Highly variable gene selection (Section 17)
    - reorder: Reordering operations (Section 18)
    - resample: Resampling operations (Section 19)
    - memory: Memory management (Section 20, 21)

Usage (Internal only):
    >>> from scl._kernel import sparse
    >>> # All functions work with raw ctypes pointers
    >>> sparse.primary_sums_csr(data_ptr, indices_ptr, indptr_ptr, None, rows, cols, nnz, output_ptr)
"""

from . import lib_loader
from . import types
from . import sparse
from . import sparse_mapped
from . import qc
from . import normalize
from . import feature
from . import stats
from . import transform
from . import algebra
from . import group
from . import scale
from . import mmd
from . import spatial
from . import hvg
from . import reorder
from . import resample
from . import memory
from . import utils
from . import mmap
from . import io
from . import sorting
from . import core

__all__ = [
    'lib_loader',
    'types',
    'sparse',
    'sparse_mapped',
    'qc',
    'normalize',
    'feature',
    'stats',
    'transform',
    'algebra',
    'group',
    'scale',
    'mmd',
    'spatial',
    'hvg',
    'reorder',
    'resample',
    'memory',
    'utils',
    'mmap',
    'io',
    'sorting',
    'core',
]

__version__ = '0.1.0'

