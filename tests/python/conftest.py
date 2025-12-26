"""
Pytest configuration and shared fixtures for SCL tests.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Try to import scl - if it fails, skip tests that require it
try:
    import scl
    from scl.sparse import Array, SclCSR, SclCSC, VirtualCSR, VirtualCSC
    from scl.sparse import float32, float64, int32, int64
    HAS_SCL = True
except ImportError as e:
    HAS_SCL = False
    SCL_IMPORT_ERROR = str(e)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def requires_scl():
    """Skip test if SCL is not available."""
    if not HAS_SCL:
        pytest.skip(f"SCL not available: {SCL_IMPORT_ERROR}")


@pytest.fixture
def small_csr_matrix(requires_scl):
    """Create a small test CSR matrix (3x4)."""
    # Create a simple 3x4 matrix:
    # [[1, 0, 2, 0],
    #  [0, 3, 0, 4],
    #  [5, 0, 0, 6]]
    data = Array.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float32)
    indices = Array.from_list([0, 2, 1, 3, 0, 3], dtype=int64)
    indptr = Array.from_list([0, 2, 4, 6], dtype=int64)
    
    return SclCSR(data, indices, indptr, shape=(3, 4))


@pytest.fixture
def small_csc_matrix(requires_scl):
    """Create a small test CSC matrix (3x4)."""
    # Same matrix as above but in CSC format
    data = Array.from_list([1.0, 5.0, 3.0, 2.0, 4.0, 6.0], dtype=float32)
    indices = Array.from_list([0, 2, 1, 0, 1, 2], dtype=int64)
    indptr = Array.from_list([0, 2, 3, 4, 6], dtype=int64)
    
    return SclCSC(data, indices, indptr, shape=(3, 4))


@pytest.fixture
def dense_matrix_small():
    """Create a small dense numpy matrix for comparison."""
    return np.array([
        [1, 0, 2, 0],
        [0, 3, 0, 4],
        [5, 0, 0, 6]
    ], dtype=np.float32)


@pytest.fixture
def random_sparse_matrix(requires_scl):
    """Create a random sparse matrix for testing."""
    np.random.seed(42)
    rows, cols = 100, 200
    density = 0.1
    nnz = int(rows * cols * density)
    
    # Generate random sparse matrix data
    data = np.random.randn(nnz).astype(np.float32)
    indices = np.random.randint(0, cols, size=nnz, dtype=np.int64)
    
    # Create indptr (row pointers)
    row_counts = np.zeros(rows, dtype=np.int64)
    row_indices = np.random.randint(0, rows, size=nnz, dtype=np.int64)
    for idx in row_indices:
        row_counts[idx] += 1
    
    indptr = np.zeros(rows + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(row_counts)
    
    # Sort by row and column for valid CSR format
    sort_idx = np.lexsort((indices, row_indices))
    indices = indices[sort_idx]
    data = data[sort_idx]
    
    data_arr = Array.from_list(data.tolist(), dtype=float32)
    indices_arr = Array.from_list(indices.tolist(), dtype=int64)
    indptr_arr = Array.from_list(indptr.tolist(), dtype=int64)
    
    return SclCSR(data_arr, indices_arr, indptr_arr, shape=(rows, cols))


# =============================================================================
# Helper Functions
# =============================================================================

def assert_array_equal(a1, a2, rtol=1e-5, atol=1e-8):
    """Assert two arrays are approximately equal."""
    if hasattr(a1, 'to_numpy'):
        a1 = a1.to_numpy()
    if hasattr(a2, 'to_numpy'):
        a2 = a2.to_numpy()
    
    np.testing.assert_allclose(a1, a2, rtol=rtol, atol=atol)


def csr_to_dense(csr, shape):
    """Convert CSR matrix to dense numpy array."""
    dense = np.zeros(shape, dtype=np.float32)
    data = csr.data.to_numpy() if hasattr(csr.data, 'to_numpy') else csr.data
    indices = csr.indices.to_numpy() if hasattr(csr.indices, 'to_numpy') else csr.indices
    indptr = csr.indptr.to_numpy() if hasattr(csr.indptr, 'to_numpy') else csr.indptr
    
    for i in range(shape[0]):
        start = indptr[i]
        end = indptr[i + 1]
        for j in range(start, end):
            dense[i, indices[j]] = data[j]
    
    return dense

