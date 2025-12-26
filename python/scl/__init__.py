"""
SCL - Sparse Computing Library

High-performance sparse matrix library with lazy loading support for
large-scale datasets.

Core Components (zero external dependencies):
    - RealArray, IndexArray, ByteArray: Memory-managed arrays
    - SclCSR: Intelligent CSR sparse matrix
    - SclCSC: Intelligent CSC sparse matrix
    - DType: Data type enumeration
    - config: Global configuration system
    - Backend: Storage backend types

Optional Components (requires numpy/scipy):
    - lazy module: LazyView, PairSparse, etc.
    - to_scipy/from_scipy conversions
"""

# Core array types
from scl.array import (
    RealArray,
    IndexArray,
    ByteArray,
    SIZEOF_REAL,
    SIZEOF_INDEX,
    SIZEOF_BYTE,
    DType,
)

# Data types
from scl._dtypes import (
    Real,
    Index,
    Byte,
    SCL_ALIGNMENT,
    validate_dtype,
)

# Core sparse types
from scl.sparse import (
    SclCSR,
    SclCSC,
    SparseBase,
    LazyLevel,
    SparseAdapter,
)

# Backend abstractions
from scl._backend import (
    Backend,
    Ownership,
    StorageInfo,
    CustomStorage,
    VirtualStorage,
    MappedStorage,
    suggest_backend,
    estimate_memory,
)

# Configuration system
from scl._config import (
    config,
    get_config,
    set_lazy,
    set_parallel,
    set_memory,
    MaterializeStrategy,
    ParallelStrategy,
    MemoryStrategy,
    NormType,
)

# FFI utilities
from scl._ffi import (
    get_lib,
    get_lib_with_signatures,
    check_error,
    SclError,
)

# Operations module
from scl import ops

# Convenience imports from ops module
from scl.ops import (
    # Normalization
    normalize,
    normalize_csc,
    # Transforms
    log1p,
    log1p_csc,
    log2p1,
    expm1,
    softmax,
    scale,
    scale_csc,
    # Filtering
    filter_threshold,
    top_k,
    # Matrix operations
    spmv,
    dot,
    gram,
    pearson,
    # Statistics
    var,
    var_csc,
    std,
    std_csc,
    sum_csc,
    mean_csc,
    dispersion,
    nnz_per_row,
    nnz_per_col,
    # Primary axis statistics
    row_sums,
    col_sums,
    row_means,
    col_means,
    row_variances,
    col_variances,
    # Statistical tests
    mwu_test,
    ttest,
    # Group statistics
    group_stats,
    count_group_sizes,
    # Feature statistics
    detection_rate,
    standard_moments,
    clipped_moments,
    # Quality Control
    compute_qc,
    # Feature Selection
    highly_variable,
    # Standardization
    standardize,
    # Resampling
    downsample,
    # Spatial
    mmd_rbf,
    morans_i,
    # Reordering
    align_secondary,
    # Utility
    copy,
    issparse,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Array types
    "RealArray",
    "IndexArray",
    "ByteArray",
    "SIZEOF_REAL",
    "SIZEOF_INDEX",
    "SIZEOF_BYTE",
    "DType",
    # Primary types
    "Real",
    "Index",
    "Byte",
    "SCL_ALIGNMENT",
    "validate_dtype",
    # Sparse types
    "SclCSR",
    "SclCSC",
    "SparseBase",
    "LazyLevel",
    "SparseAdapter",
    # Backend
    "Backend",
    "Ownership",
    "StorageInfo",
    "CustomStorage",
    "VirtualStorage",
    "MappedStorage",
    "suggest_backend",
    "estimate_memory",
    # Configuration
    "config",
    "get_config",
    "set_lazy",
    "set_parallel",
    "set_memory",
    "MaterializeStrategy",
    "ParallelStrategy",
    "MemoryStrategy",
    "NormType",
    # FFI
    "get_lib",
    "get_lib_with_signatures",
    "check_error",
    "SclError",
    # Ops module
    "ops",
    # Operations - Normalization
    "normalize",
    "normalize_csc",
    # Operations - Transforms
    "log1p",
    "log1p_csc",
    "log2p1",
    "expm1",
    "softmax",
    "scale",
    "scale_csc",
    # Operations - Filtering
    "filter_threshold",
    "top_k",
    # Operations - Matrix
    "spmv",
    "dot",
    "gram",
    "pearson",
    # Operations - Statistics
    "var",
    "var_csc",
    "std",
    "std_csc",
    "sum_csc",
    "mean_csc",
    "dispersion",
    "nnz_per_row",
    "nnz_per_col",
    # Operations - Primary axis statistics
    "row_sums",
    "col_sums",
    "row_means",
    "col_means",
    "row_variances",
    "col_variances",
    # Operations - Statistical tests
    "mwu_test",
    "ttest",
    # Operations - Group statistics
    "group_stats",
    "count_group_sizes",
    # Operations - Feature statistics
    "detection_rate",
    "standard_moments",
    "clipped_moments",
    # Operations - Quality Control
    "compute_qc",
    # Operations - Feature Selection
    "highly_variable",
    # Operations - Standardization
    "standardize",
    # Operations - Resampling
    "downsample",
    # Operations - Spatial
    "mmd_rbf",
    "morans_i",
    # Operations - Reordering
    "align_secondary",
    # Operations - Utility
    "copy",
    "issparse",
]


# Lazy import for optional numpy-dependent modules
def __getattr__(name):
    """Lazy import for optional modules."""
    if name in ("LazyView", "LazyReorder", "PairSparse", "LazyVStack", "hstack", "vstack"):
        from scl import lazy
        return getattr(lazy, name)
    raise AttributeError(f"module 'scl' has no attribute '{name}'")
