"""
SCL Python Bindings
===================

Auto-generated ctypes bindings for the SCL native library.

Usage:
    from scl._bindings import algebra, neighbors, ...
    
    # Or import specific functions
    from scl._bindings.algebra import scl_matrix_multiply
"""

from __future__ import annotations

from ._loader import get_library, get_available_precisions

from . import algebra
from . import alignment
from . import annotation
from . import association
from . import bbknn
from . import centrality
from . import clonotype
from . import coexpression
from . import communication
from . import comparison
from . import components
from . import core
from . import correlation
from . import diffusion
from . import doublet
from . import enrichment
from . import entropy
from . import feature
from . import gnn
from . import gram
from . import grn
from . import group
from . import hotspot
from . import hvg
from . import impute
from . import kernel
from . import leiden
from . import lineage
from . import log1p
from . import louvain
from . import markers
from . import merge
from . import metrics
from . import mmd
from . import multiple_testing
from . import mwu
from . import neighbors
from . import niche
from . import normalize
from . import outlier
from . import permutation
from . import projection
from . import propagation
from . import pseudotime
from . import qc
from . import reorder
from . import resample
from . import sampling
from . import scale
from . import scoring
from . import slice
from . import softmax
from . import sparse_kernel
from . import sparse_opt
from . import spatial
from . import spatial_pattern
from . import stat
from . import state
from . import subpopulation
from . import tissue
from . import transition
from . import ttest
from . import velocity

__all__ = [
    "get_library",
    "get_available_precisions",
    "algebra",
    "alignment",
    "annotation",
    "association",
    "bbknn",
    "centrality",
    "clonotype",
    "coexpression",
    "communication",
    "comparison",
    "components",
    "core",
    "correlation",
    "diffusion",
    "doublet",
    "enrichment",
    "entropy",
    "feature",
    "gnn",
    "gram",
    "grn",
    "group",
    "hotspot",
    "hvg",
    "impute",
    "kernel",
    "leiden",
    "lineage",
    "log1p",
    "louvain",
    "markers",
    "merge",
    "metrics",
    "mmd",
    "multiple_testing",
    "mwu",
    "neighbors",
    "niche",
    "normalize",
    "outlier",
    "permutation",
    "projection",
    "propagation",
    "pseudotime",
    "qc",
    "reorder",
    "resample",
    "sampling",
    "scale",
    "scoring",
    "slice",
    "softmax",
    "sparse_kernel",
    "sparse_opt",
    "spatial",
    "spatial_pattern",
    "stat",
    "state",
    "subpopulation",
    "tissue",
    "transition",
    "ttest",
    "velocity",
]
