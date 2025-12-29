#!/usr/bin/env python3
"""
Comprehensive test suite for SCL Python core module.

Tests:
- Error handling alignment with C API
- Automatic library selection
- CSR/CSC matrix operations
- Ownership semantics
- Dense matrix (unsafe) operations
- Lazy scipy import
"""

import sys
import numpy as np
import pytest

# Add python directory to path
sys.path.insert(0, 'python')

import scl
from scl import CsrMatrix, CscMatrix, DenseMatrix, SCLError


class TestErrorHandling:
    """Test error handling and code alignment."""
    
    def test_error_codes_aligned(self):
        """Error codes must match C API."""
        assert SCLError.ERROR_NULL_POINTER == 4
        assert SCLError.ERROR_INVALID_ARGUMENT == 10
        assert SCLError.ERROR_DIMENSION_MISMATCH == 11
        assert SCLError.OK == 0
    
    def test_error_message(self):
        """Error messages should be informative."""
        err = SCLError(10, "Test error")
        assert "Test error" in str(err)
        assert err.code == 10


class TestConfiguration:
    """Test global configuration and library selection."""
    
    def test_default_precision(self):
        """Default should be float64 + int64."""
        real, index = scl.get_precision()
        assert real == scl.RealType.FLOAT64
        assert index == scl.IndexType.INT64
    
    def test_set_precision(self):
        """Should be able to change precision."""
        scl.set_precision(real='float32', index='int32')
        real, index = scl.get_precision()
        assert real == scl.RealType.FLOAT32
        assert index == scl.IndexType.INT32
        # Reset
        scl.set_precision(real='float64', index='int64')
    
    def test_library_lazy_loading(self):
        """Library should be lazy-loaded."""
        lib = scl.get_default_library()
        assert lib.variant == "f64_i64"


class TestCsrMatrix:
    """Test CSR matrix operations."""
    
    def test_wrap(self):
        """Test zero-copy wrap."""
        indptr = np.array([0, 2, 3, 6], dtype=np.int64)
        indices = np.array([0, 2, 2, 0, 1, 2], dtype=np.int64)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        
        mat = CsrMatrix.wrap(indptr, indices, data, (3, 3))
        assert mat.shape == (3, 3)
        assert mat.nnz == 6
        assert mat.format == 'csr'
        assert mat.is_view
    
    def test_copy_from_scipy(self):
        """Test deep copy from SciPy."""
        from scipy.sparse import csr_matrix
        
        scipy_mat = csr_matrix(([1, 2, 3], ([0, 0, 1], [0, 2, 2])), shape=(3, 3))
        mat = CsrMatrix.copy(scipy_mat)
        
        assert mat.shape == (3, 3)
        assert mat.nnz == 3
        assert mat.is_owned
    
    def test_row_slicing(self):
        """Test row slicing (zero-copy)."""
        from scipy.sparse import random as sp_random
        
        scipy_mat = sp_random(100, 50, density=0.1, format='csr', dtype=np.float64)
        mat = CsrMatrix.copy(scipy_mat)
        
        # Slice
        rows = mat[10:20]
        assert rows.shape == (10, 50)
        assert rows.is_view
    
    def test_transpose(self):
        """Test CSR -> CSC transpose."""
        from scipy.sparse import random as sp_random
        
        scipy_mat = sp_random(1000, 500, density=0.05, format='csr', dtype=np.float64)
        csr = CsrMatrix.copy(scipy_mat)
        
        csc = csr.transpose()
        assert isinstance(csc, CscMatrix)
        assert csc.shape == (500, 1000)
        assert csc.nnz == csr.nnz
    
    def test_clone(self):
        """Test deep copy."""
        from scipy.sparse import csr_matrix
        
        scipy_mat = csr_matrix(([1, 2, 3], ([0, 0, 1], [0, 2, 2])), shape=(3, 3))
        mat = CsrMatrix.copy(scipy_mat)
        cloned = mat.clone()
        
        assert cloned.shape == mat.shape
        assert cloned.nnz == mat.nnz
        assert cloned.is_owned
    
    def test_vstack(self):
        """Test vertical stacking."""
        from scipy.sparse import csr_matrix
        
        mat1 = CsrMatrix.copy(csr_matrix(([1, 2], ([0, 1], [0, 1])), shape=(2, 3)))
        mat2 = CsrMatrix.copy(csr_matrix(([3, 4], ([0, 1], [1, 2])), shape=(2, 3)))
        
        stacked = CsrMatrix.vstack([mat1, mat2])
        assert stacked.shape == (4, 3)


class TestCscMatrix:
    """Test CSC matrix operations."""
    
    def test_copy_from_scipy(self):
        """Test deep copy from SciPy CSC."""
        from scipy.sparse import csc_matrix
        
        scipy_mat = csc_matrix(([1, 2, 3], ([0, 0, 1], [0, 2, 2])), shape=(3, 3))
        mat = CscMatrix.copy(scipy_mat)
        
        assert mat.shape == (3, 3)
        assert mat.format == 'csc'
    
    def test_transpose(self):
        """Test CSC -> CSR transpose."""
        from scipy.sparse import random as sp_random
        
        scipy_mat = sp_random(1000, 500, density=0.05, format='csc', dtype=np.float64)
        csc = CscMatrix.copy(scipy_mat)
        
        csr = csc.transpose()
        assert isinstance(csr, CsrMatrix)
        assert csr.shape == (500, 1000)
    
    def test_hstack(self):
        """Test horizontal stacking."""
        from scipy.sparse import csc_matrix
        
        mat1 = CscMatrix.copy(csc_matrix(([1, 2], ([0, 1], [0, 1])), shape=(3, 2)))
        mat2 = CscMatrix.copy(csc_matrix(([3, 4], ([0, 1], [0, 1])), shape=(3, 2)))
        
        stacked = CscMatrix.hstack([mat1, mat2])
        assert stacked.shape == (3, 4)


class TestDenseMatrix:
    """Test dense matrix (unsafe) operations."""
    
    def test_wrap(self):
        """Test zero-copy wrap."""
        arr = np.random.randn(10, 5).astype(np.float64)
        
        with DenseMatrix.wrap(arr) as dense:
            assert dense.shape == (10, 5)
            assert dense.dtype == np.float64
    
    def test_1d_array(self):
        """Test 1D array wrapping."""
        arr = np.random.randn(100).astype(np.float64)
        
        with DenseMatrix.wrap(arr) as dense:
            assert dense.shape == (100, 1)
    
    def test_element_access(self):
        """Test element get/set."""
        arr = np.zeros((5, 3), dtype=np.float64)
        
        with DenseMatrix.wrap(arr) as dense:
            dense.set(2, 1, 42.0)
            val = dense.get(2, 1)
            assert val == 42.0
            assert arr[2, 1] == 42.0  # Verify zero-copy


class TestOwnership:
    """Test ownership semantics."""
    
    def test_owned_matrix(self):
        """Owned matrix should be independent."""
        from scipy.sparse import csr_matrix
        
        scipy_mat = csr_matrix(([1, 2], ([0, 1], [0, 1])), shape=(2, 2))
        mat = CsrMatrix.copy(scipy_mat)
        
        assert mat.is_owned
        assert not mat.is_view
    
    def test_view_matrix(self):
        """View matrix should share data."""
        indptr = np.array([0, 2, 3], dtype=np.int64)
        indices = np.array([0, 1, 1], dtype=np.int64)
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        
        mat = CsrMatrix.wrap(indptr, indices, data, (2, 2))
        
        assert mat.is_view
        assert not mat.is_owned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

