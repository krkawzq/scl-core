"""
Pytest configuration and shared fixtures for SCL tests.

Updated for SCL v0.2.0 with smart backend management.
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
    from scl.sparse import (
        Array, SclCSR, SclCSC,
        VirtualCSR, VirtualCSC,  # Aliases for backward compatibility
        Backend, Ownership,
        float32, float64, int32, int64,
        vstack_csr, hstack_csc,
        from_scipy,
    )
    HAS_SCL = True
except ImportError as e:
    HAS_SCL = False
    SCL_IMPORT_ERROR = str(e)


# Try to import scipy
try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def requires_scl():
    """Skip test if SCL is not available."""
    if not HAS_SCL:
        pytest.skip(f"SCL not available: {SCL_IMPORT_ERROR}")


@pytest.fixture(scope="session")
def requires_scipy():
    """Skip test if scipy is not available."""
    if not HAS_SCIPY:
        pytest.skip("scipy not available")


@pytest.fixture
def small_csr_matrix(requires_scl):
    """Create a small test CSR matrix (3x4).
    
    Matrix:
    [[1, 0, 2, 0],
     [0, 3, 0, 4],
     [5, 0, 0, 6]]
    """
    data = Array.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float32)
    indices = Array.from_list([0, 2, 1, 3, 0, 3], dtype=int64)
    indptr = Array.from_list([0, 2, 4, 6], dtype=int64)
    
    return SclCSR(data, indices, indptr, shape=(3, 4))


@pytest.fixture
def small_csc_matrix(requires_scl):
    """Create a small test CSC matrix (3x4).
    
    Same matrix as small_csr but in CSC format.
    """
    data = Array.from_list([1.0, 5.0, 3.0, 2.0, 4.0, 6.0], dtype=float32)
    indices = Array.from_list([0, 2, 1, 0, 1, 2], dtype=int64)
    indptr = Array.from_list([0, 2, 3, 4, 6], dtype=int64)
    
    return SclCSC(data, indices, indptr, shape=(3, 4))


@pytest.fixture
def small_csr_from_dense(requires_scl):
    """Create CSR from dense for easier testing."""
    return SclCSR.from_dense([
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0, 4.0],
        [5.0, 0.0, 0.0, 6.0]
    ], dtype='float32')


@pytest.fixture
def small_csc_from_dense(requires_scl):
    """Create CSC from dense for easier testing."""
    return SclCSC.from_dense([
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0, 4.0],
        [5.0, 0.0, 0.0, 6.0]
    ], dtype='float32')


@pytest.fixture
def dense_matrix_small():
    """Create a small dense numpy matrix for comparison."""
    return np.array([
        [1, 0, 2, 0],
        [0, 3, 0, 4],
        [5, 0, 0, 6]
    ], dtype=np.float32)


@pytest.fixture
def scipy_csr_matrix(requires_scipy):
    """Create a scipy CSR matrix for interop testing."""
    return sp.csr_matrix([
        [1, 0, 2, 0],
        [0, 3, 0, 4],
        [5, 0, 0, 6]
    ], dtype=np.float32)


@pytest.fixture
def scipy_csc_matrix(requires_scipy):
    """Create a scipy CSC matrix for interop testing."""
    return sp.csc_matrix([
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
    if hasattr(a1, 'tolist'):
        a1 = np.array(a1.tolist())
    if hasattr(a2, 'tolist'):
        a2 = np.array(a2.tolist())
    
    np.testing.assert_allclose(a1, a2, rtol=rtol, atol=atol)


def assert_matrices_equal(mat1, mat2, rtol=1e-5):
    """Assert two sparse matrices have equal values."""
    assert mat1.shape == mat2.shape
    assert mat1.nnz == mat2.nnz
    
    scipy1 = mat1.to_scipy()
    scipy2 = mat2.to_scipy()
    
    np.testing.assert_allclose(scipy1.toarray(), scipy2.toarray(), rtol=rtol)


def csr_to_dense(csr, shape=None):
    """Convert CSR matrix to dense numpy array."""
    if shape is None:
        shape = csr.shape
    
    dense = np.zeros(shape, dtype=np.float32)
    
    for i in range(shape[0]):
        indices, values = csr.get_row(i)
        for k in range(len(indices)):
            dense[i, indices[k]] = values[k]
    
    return dense
