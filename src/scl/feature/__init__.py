"""
SCL Feature Module.

This module provides feature analysis and selection operations for sparse
matrices, essential for dimension reduction and quality control in
single-cell analysis.

Submodules:
    - selection: Highly variable gene selection and related methods
    - qc: Quality control metrics and filtering

Feature selection is crucial for:
    - Reducing noise from lowly-expressed genes
    - Focusing on biologically informative features
    - Improving downstream analysis (clustering, trajectory)

Example:
    >>> import scl.feature as feat
    >>> from scl import SclCSR
    >>>
    >>> # Quality control
    >>> counts, sums = feat.compute_qc(mat)
    >>>
    >>> # Highly variable gene selection
    >>> hvg_indices = feat.highly_variable(mat, n_top=2000)
"""

from scl.feature.selection import (
    highly_variable,
    dispersion,
    detection_rate,
)

from scl.feature.qc import (
    compute_qc,
    standard_moments,
    clipped_moments,
)

__all__ = [
    # Selection
    "highly_variable",
    "dispersion",
    "detection_rate",
    # QC
    "compute_qc",
    "standard_moments",
    "clipped_moments",
]
