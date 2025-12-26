"""
SCL Spatial Module.

This module provides spatial analysis operations for sparse matrices,
particularly useful for spatial transcriptomics data analysis.

Submodules:
    - autocorrelation: Moran's I and other spatial autocorrelation metrics

Spatial analysis methods help identify:
    - Spatially variable genes
    - Spatial patterns in expression
    - Domain/region identification

Example:
    >>> import scl.spatial as spatial
    >>> from scl import SclCSC
    >>>
    >>> # Compute spatial autocorrelation
    >>> morans = spatial.morans_i(expression, spatial_weights)
"""

from scl.spatial.autocorrelation import (
    morans_i,
    mmd_rbf,
)

__all__ = [
    "morans_i",
    "mmd_rbf",
]
