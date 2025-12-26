"""
SCL Preprocessing Module.

This module provides data preprocessing operations for sparse matrices,
essential for preparing single-cell data for downstream analysis.

Submodules:
    - normalize: L1/L2/Max normalization
    - scale: Row/column scaling and standardization
    - filter: Thresholding, top-k selection, downsampling

All functions support multiple input formats through the overload dispatch
system, including native SCL types, scipy sparse matrices, and numpy arrays.

Typical Preprocessing Pipeline:
    1. Quality control filtering (feature module)
    2. Normalization (this module)
    3. Log transformation (math module)
    4. Scaling/Standardization (this module)
    5. Feature selection (feature module)

Example:
    >>> import scl.preprocessing as pp
    >>> from scl import SclCSR
    >>>
    >>> # Typical single-cell preprocessing
    >>> mat = SclCSR.from_scipy(raw_counts)
    >>> mat = pp.normalize(mat, norm="l1", axis=1)  # Library size normalization
    >>> mat = pp.scale(mat, target_sum=10000)       # Scale to 10k counts
"""

from scl.preprocessing.normalize import (
    normalize,
    standardize,
)

from scl.preprocessing.scale import (
    scale,
)

from scl.preprocessing.filter import (
    filter_threshold,
    top_k,
    downsample,
)

from scl.preprocessing.transform import (
    log1p,
    softmax,
)

from scl.preprocessing.resample import (
    downsample_counts,
)

__all__ = [
    # Normalization
    "normalize",
    "standardize",
    # Scaling
    "scale",
    # Filtering
    "filter_threshold",
    "top_k",
    "downsample",
    # Transforms
    "log1p",
    "softmax",
    # Resampling
    "downsample_counts",
]
