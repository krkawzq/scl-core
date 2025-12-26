"""
Tests for SclCSR and SclCSC matrix classes.
"""

import pytest
import numpy as np
from scl.sparse import SclCSR, SclCSC, Array, from_list, zeros
from scl.sparse import float32, float64, int64


class TestSclCSRCreation:
    """Test SclCSR matrix creation."""
    
    def test_create_from_arrays(self, requires_scl):
        """Test creating SclCSR from arrays."""
        data = from_list([1.0, 2.0, 3.0], dtype=float32)
        indices = from_list([0, 2, 1], dtype=int64)
        indptr = from_list([0, 2, 3, 3], dtype=int64)
        
        mat = SclCSR(data, indices, indptr, shape=(3, 4))
        assert mat.shape == (3, 4)
        assert mat.nnz == 3
        assert mat.rows == 3
        assert mat.cols == 4
    
    def test_create_empty(self, requires_scl):
        """Test creating empty matrix."""
        mat = SclCSR.empty(10, 20, 50, dtype='float32')
        assert mat.shape == (10, 20)
        assert mat.nnz == 0  # Empty, but allocated for 50
        assert mat.dtype == 'float32'
    
    def test_create_zeros(self, requires_scl):
        """Test creating zero matrix."""
        mat = SclCSR.zeros(10, 20, dtype='float32')
        assert mat.shape == (10, 20)
        assert mat.nnz == 0
        assert mat.dtype == 'float32'
    
    def test_create_from_dense(self, requires_scl):
        """Test creating from dense 2D list."""
        dense = [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [5.0, 0.0, 0.0, 6.0]
        ]
        mat = SclCSR.from_dense(dense, dtype='float32')
        assert mat.shape == (3, 4)
        assert mat.nnz == 6
    
    def test_create_from_scipy(self, requires_scl):
        """Test creating from scipy matrix."""
        try:
            import scipy.sparse as sp
        except ImportError:
            pytest.skip("scipy not available")
        
        scipy_mat = sp.csr_matrix([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
        mat = SclCSR.from_scipy(scipy_mat)
        assert mat.shape == (2, 3)
        assert mat.nnz == 3


class TestSclCSRProperties:
    """Test SclCSR properties."""
    
    def test_shape_property(self, small_csr_matrix):
        """Test shape property."""
        assert small_csr_matrix.shape == (3, 4)
    
    def test_nnz_property(self, small_csr_matrix):
        """Test nnz property."""
        assert small_csr_matrix.nnz == 6
    
    def test_rows_cols_properties(self, small_csr_matrix):
        """Test rows and cols properties."""
        assert small_csr_matrix.rows == 3
        assert small_csr_matrix.cols == 4
    
    def test_dtype_property(self, small_csr_matrix):
        """Test dtype property."""
        assert small_csr_matrix.dtype == 'float32'
    
    def test_row_lengths(self, small_csr_matrix):
        """Test row_lengths array."""
        row_lengths = small_csr_matrix.row_lengths
        assert row_lengths.size == 3
        # First row has 2 non-zeros, second has 2, third has 2
        assert row_lengths[0] == 2
        assert row_lengths[1] == 2
        assert row_lengths[2] == 2


class TestSclCSRValidation:
    """Test SclCSR validation."""
    
    def test_invalid_shape(self, requires_scl):
        """Test invalid shape."""
        data = from_list([1.0], dtype=float32)
        indices = from_list([0], dtype=int64)
        indptr = from_list([0, 1], dtype=int64)
        
        with pytest.raises(ValueError):
            SclCSR(data, indices, indptr, shape=(-1, 10))
    
    def test_indptr_size_mismatch(self, requires_scl):
        """Test indptr size mismatch."""
        data = from_list([1.0], dtype=float32)
        indices = from_list([0], dtype=int64)
        indptr = from_list([0, 1, 2], dtype=int64)  # Wrong size for 3 rows
        
        with pytest.raises(ValueError):
            SclCSR(data, indices, indptr, shape=(3, 10))
    
    def test_indices_size_mismatch(self, requires_scl):
        """Test indices size mismatch."""
        data = from_list([1.0, 2.0], dtype=float32)
        indices = from_list([0], dtype=int64)  # Wrong size
        indptr = from_list([0, 2], dtype=int64)
        
        with pytest.raises(ValueError):
            SclCSR(data, indices, indptr, shape=(1, 10))
    
    def test_invalid_dtype(self, requires_scl):
        """Test invalid dtype for data."""
        data = from_list([1, 2, 3], dtype='int32')  # Must be float
        indices = from_list([0, 1, 2], dtype=int64)
        indptr = from_list([0, 3], dtype=int64)
        
        with pytest.raises(TypeError):
            SclCSR(data, indices, indptr, shape=(1, 10))


class TestSclCSROperations:
    """Test SclCSR operations."""
    
    def test_to_scipy(self, small_csr_matrix):
        """Test converting to scipy matrix."""
        try:
            import scipy.sparse as sp
        except ImportError:
            pytest.skip("scipy not available")
        
        scipy_mat = small_csr_matrix.to_scipy()
        assert isinstance(scipy_mat, sp.csr_matrix)
        assert scipy_mat.shape == (3, 4)
        assert scipy_mat.nnz == 6
    
    def test_get_row(self, small_csr_matrix):
        """Test getting a row."""
        # First row should be [1, 0, 2, 0]
        # get_row returns (indices, values) tuple
        indices, values = small_csr_matrix.get_row(0)
        assert len(indices) == 2  # 2 non-zeros in first row
        assert len(values) == 2
        
        # Check values
        assert values[0] == pytest.approx(1.0)  # data[0]
        assert values[1] == pytest.approx(2.0)  # data[1]
        
        # Or use get_row_dense for dense representation
        row0_dense = small_csr_matrix.get_row_dense(0)
        assert len(row0_dense) == 4
        assert row0_dense[0] == pytest.approx(1.0)
        assert row0_dense[2] == pytest.approx(2.0)
    
    def test_get_row_slice(self, small_csr_matrix):
        """Test getting row slice."""
        # Get first two rows
        try:
            row_slice = small_csr_matrix.get_row_slice(0, 2)
            assert row_slice.shape[0] == 2
            assert row_slice.shape[1] == 4
        except (AttributeError, NotImplementedError):
            pytest.skip("get_row_slice not implemented")


class TestSclCSC:
    """Test SclCSC matrix class."""
    
    def test_create_csc(self, requires_scl):
        """Test creating SclCSC matrix."""
        data = from_list([1.0, 2.0, 3.0], dtype=float32)
        indices = from_list([0, 2, 1], dtype=int64)
        indptr = from_list([0, 1, 2, 3, 3], dtype=int64)
        
        mat = SclCSC(data, indices, indptr, shape=(3, 4))
        assert mat.shape == (3, 4)
        assert mat.nnz == 3
    
    def test_create_csc_empty(self, requires_scl):
        """Test creating empty CSC matrix."""
        mat = SclCSC.empty(10, 20, 50, dtype='float32')
        assert mat.shape == (10, 20)
    
    def test_create_csc_zeros(self, requires_scl):
        """Test creating zero CSC matrix."""
        mat = SclCSC.zeros(10, 20, dtype='float32')
        assert mat.shape == (10, 20)
        assert mat.nnz == 0
    
    def test_csc_to_scipy(self, small_csc_matrix):
        """Test converting CSC to scipy."""
        try:
            import scipy.sparse as sp
        except ImportError:
            pytest.skip("scipy not available")
        
        scipy_mat = small_csc_matrix.to_scipy()
        assert isinstance(scipy_mat, sp.csc_matrix)
        assert scipy_mat.shape == (3, 4)


class TestMatrixConversion:
    """Test matrix conversion methods."""
    
    def test_csr_to_csc(self, small_csr_matrix):
        """Test converting CSR to CSC."""
        try:
            csc = small_csr_matrix.tocsc()
            assert csc.shape == small_csr_matrix.shape
            assert csc.nnz == small_csr_matrix.nnz
        except (AttributeError, NotImplementedError):
            pytest.skip("tocsc not implemented")
    
    def test_csc_to_csr(self, small_csc_matrix):
        """Test converting CSC to CSR."""
        try:
            csr = small_csc_matrix.tocsr()
            assert csr.shape == small_csc_matrix.shape
            assert csr.nnz == small_csc_matrix.nnz
        except (AttributeError, NotImplementedError):
            pytest.skip("tocsr not implemented")

