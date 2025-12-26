"""
Tests for SclCSR and SclCSC smart matrix classes.

Updated for SCL v0.2.0 with backend management.
"""

import pytest
import numpy as np

try:
    from scl.sparse import (
        SclCSR, SclCSC, Array, 
        Backend, Ownership,
        from_list, zeros,
        float32, float64, int64,
    )
    HAS_SCL = True
except ImportError:
    HAS_SCL = False

try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# SclCSR Creation Tests
# =============================================================================

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
        assert mat.backend == Backend.CUSTOM
        assert mat.ownership == Ownership.OWNED
    
    def test_create_empty(self, requires_scl):
        """Test creating empty matrix."""
        mat = SclCSR.empty(10, 20, 50, dtype='float32')
        assert mat.shape == (10, 20)
        assert mat.nnz == 50
        assert len(mat.data) == 50
        assert mat.dtype == 'float32'
        assert mat.backend == Backend.CUSTOM
    
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
        assert mat.backend == Backend.CUSTOM
        assert mat.ownership == Ownership.OWNED
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_create_from_scipy_borrowed(self, requires_scl):
        """Test creating from scipy matrix (borrowed mode)."""
        scipy_mat = sp.csr_matrix([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
        mat = SclCSR.from_scipy(scipy_mat, copy=False)
        
        assert mat.shape == (2, 3)
        assert mat.nnz == 3
        assert mat.backend == Backend.CUSTOM
        assert mat.ownership == Ownership.BORROWED
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_create_from_scipy_copy(self, requires_scl):
        """Test creating from scipy matrix (copy mode)."""
        scipy_mat = sp.csr_matrix([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
        mat = SclCSR.from_scipy(scipy_mat, copy=True)
        
        assert mat.shape == (2, 3)
        assert mat.nnz == 3
        assert mat.backend == Backend.CUSTOM
        assert mat.ownership == Ownership.OWNED


# =============================================================================
# SclCSR Properties Tests
# =============================================================================

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
        assert row_lengths[0] == 2
        assert row_lengths[1] == 2
        assert row_lengths[2] == 2
    
    def test_backend_property(self, small_csr_matrix):
        """Test backend property."""
        assert small_csr_matrix.backend == Backend.CUSTOM
    
    def test_ownership_property(self, small_csr_matrix):
        """Test ownership property."""
        assert small_csr_matrix.ownership == Ownership.OWNED
    
    def test_is_view_property(self, small_csr_matrix):
        """Test is_view property."""
        assert small_csr_matrix.is_view == False


# =============================================================================
# SclCSR Validation Tests
# =============================================================================

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
        indptr = from_list([0, 1, 2], dtype=int64)
        
        with pytest.raises(ValueError):
            SclCSR(data, indices, indptr, shape=(3, 10))
    
    def test_indices_size_mismatch(self, requires_scl):
        """Test indices size mismatch."""
        data = from_list([1.0, 2.0], dtype=float32)
        indices = from_list([0], dtype=int64)
        indptr = from_list([0, 2], dtype=int64)
        
        with pytest.raises(ValueError):
            SclCSR(data, indices, indptr, shape=(1, 10))
    
    def test_invalid_dtype(self, requires_scl):
        """Test invalid dtype for data."""
        data = from_list([1, 2, 3], dtype='int32')
        indices = from_list([0, 1, 2], dtype=int64)
        indptr = from_list([0, 3], dtype=int64)
        
        with pytest.raises(TypeError):
            SclCSR(data, indices, indptr, shape=(1, 10))


# =============================================================================
# SclCSR Operations Tests
# =============================================================================

class TestSclCSROperations:
    """Test SclCSR operations."""
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_to_scipy(self, small_csr_from_dense):
        """Test converting to scipy matrix."""
        scipy_mat = small_csr_from_dense.to_scipy()
        assert isinstance(scipy_mat, sp.csr_matrix)
        assert scipy_mat.shape == (3, 4)
        assert scipy_mat.nnz == 6
    
    def test_get_row(self, small_csr_from_dense):
        """Test getting a row."""
        indices, values = small_csr_from_dense.get_row(0)
        assert len(indices) == 2
        assert len(values) == 2
        assert values[0] == pytest.approx(1.0)
        assert values[1] == pytest.approx(2.0)
    
    def test_get_row_dense(self, small_csr_from_dense):
        """Test getting row as dense array."""
        row0_dense = small_csr_from_dense.get_row_dense(0)
        assert len(row0_dense) == 4
        assert row0_dense[0] == pytest.approx(1.0)
        assert row0_dense[1] == pytest.approx(0.0)
        assert row0_dense[2] == pytest.approx(2.0)
        assert row0_dense[3] == pytest.approx(0.0)
    
    def test_getitem_scalar(self, small_csr_from_dense):
        """Test mat[i, j] indexing."""
        assert small_csr_from_dense[0, 0] == pytest.approx(1.0)
        assert small_csr_from_dense[0, 1] == pytest.approx(0.0)
        assert small_csr_from_dense[0, 2] == pytest.approx(2.0)
        assert small_csr_from_dense[1, 1] == pytest.approx(3.0)
        assert small_csr_from_dense[2, 0] == pytest.approx(5.0)
    
    def test_getitem_row(self, small_csr_from_dense):
        """Test mat[i, :] indexing."""
        row = small_csr_from_dense[1, :]
        assert len(row) == 4
        assert row[1] == pytest.approx(3.0)
        assert row[3] == pytest.approx(4.0)
    
    def test_getitem_col(self, small_csr_from_dense):
        """Test mat[:, j] indexing."""
        col = small_csr_from_dense[:, 0]
        assert len(col) == 3
        assert col[0] == pytest.approx(1.0)
        assert col[1] == pytest.approx(0.0)
        assert col[2] == pytest.approx(5.0)
    
    def test_slice_rows(self, small_csr_from_dense):
        """Test row slicing."""
        sub = small_csr_from_dense[0:2, :]
        assert sub.shape == (2, 4)
        assert sub.nnz == 4
    
    def test_copy(self, small_csr_from_dense):
        """Test copy operation."""
        copy = small_csr_from_dense.copy()
        assert copy.shape == small_csr_from_dense.shape
        assert copy.nnz == small_csr_from_dense.nnz
        assert copy.backend == Backend.CUSTOM
        assert copy.ownership == Ownership.OWNED
        
        # Verify it's a deep copy
        copy.data[0] = 999.0
        assert small_csr_from_dense.data[0] != 999.0


# =============================================================================
# SclCSC Tests
# =============================================================================

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
        assert mat.backend == Backend.CUSTOM
    
    def test_create_csc_empty(self, requires_scl):
        """Test creating empty CSC matrix."""
        mat = SclCSC.empty(10, 20, 50, dtype='float32')
        assert mat.shape == (10, 20)
    
    def test_create_csc_zeros(self, requires_scl):
        """Test creating zero CSC matrix."""
        mat = SclCSC.zeros(10, 20, dtype='float32')
        assert mat.shape == (10, 20)
        assert mat.nnz == 0
    
    def test_create_csc_from_dense(self, requires_scl):
        """Test creating CSC from dense."""
        mat = SclCSC.from_dense([[1, 0, 2], [0, 3, 0]], dtype='float32')
        assert mat.shape == (2, 3)
        assert mat.nnz == 3
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_csc_to_scipy(self, small_csc_from_dense):
        """Test converting CSC to scipy."""
        scipy_mat = small_csc_from_dense.to_scipy()
        assert isinstance(scipy_mat, sp.csc_matrix)
        assert scipy_mat.shape == (3, 4)
    
    def test_get_col(self, small_csc_from_dense):
        """Test getting a column."""
        indices, values = small_csc_from_dense.get_col(0)
        assert len(indices) == 2
        assert len(values) == 2
    
    def test_get_col_dense(self, small_csc_from_dense):
        """Test getting column as dense array."""
        col0 = small_csc_from_dense.get_col_dense(0)
        assert len(col0) == 3
        assert col0[0] == pytest.approx(1.0)
        assert col0[2] == pytest.approx(5.0)


# =============================================================================
# Matrix Conversion Tests
# =============================================================================

class TestMatrixConversion:
    """Test matrix conversion methods."""
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_csr_to_csc(self, small_csr_from_dense):
        """Test converting CSR to CSC."""
        csc = small_csr_from_dense.tocsc()
        assert isinstance(csc, SclCSC)
        assert csc.shape == small_csr_from_dense.shape
        assert csc.nnz == small_csr_from_dense.nnz
        assert csc.backend == Backend.CUSTOM
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_csc_to_csr(self, small_csc_from_dense):
        """Test converting CSC to CSR."""
        csr = small_csc_from_dense.tocsr()
        assert isinstance(csr, SclCSR)
        assert csr.shape == small_csc_from_dense.shape
        assert csr.nnz == small_csc_from_dense.nnz
    
    def test_to_owned(self, small_csr_from_dense):
        """Test to_owned conversion."""
        owned = small_csr_from_dense.to_owned()
        assert owned.backend == Backend.CUSTOM
        assert owned.ownership == Ownership.OWNED


# =============================================================================
# Matrix Info Tests
# =============================================================================

class TestMatrixInfo:
    """Test matrix info methods."""
    
    def test_repr(self, small_csr_from_dense):
        """Test __repr__ method."""
        repr_str = repr(small_csr_from_dense)
        assert 'SclCSR' in repr_str
        assert '3, 4' in repr_str or '(3, 4)' in repr_str
    
    def test_info(self, small_csr_from_dense):
        """Test info() method."""
        info_str = small_csr_from_dense.info()
        assert 'SclCSR' in info_str
        assert 'shape' in info_str
        assert 'backend' in info_str
        assert 'ownership' in info_str


# =============================================================================
# Statistical Methods Tests
# =============================================================================

class TestCSRStatisticalMethods:
    """Test SclCSR statistical methods (sum, mean, min, max)."""
    
    def test_sum_total(self, small_csr_from_dense):
        """Test total sum of matrix."""
        # Matrix: [[1, 0, 2, 0], [0, 3, 0, 4], [5, 0, 0, 6]]
        # Sum: 1+2+3+4+5+6 = 21
        total = small_csr_from_dense.sum()
        assert total == pytest.approx(21.0)
    
    def test_sum_axis1_row_sums(self, small_csr_from_dense):
        """Test row sums (axis=1)."""
        row_sums = small_csr_from_dense.sum(axis=1)
        assert len(row_sums) == 3
        assert row_sums[0] == pytest.approx(3.0)   # 1+2
        assert row_sums[1] == pytest.approx(7.0)   # 3+4
        assert row_sums[2] == pytest.approx(11.0)  # 5+6
    
    def test_sum_axis0_col_sums(self, small_csr_from_dense):
        """Test column sums (axis=0)."""
        col_sums = small_csr_from_dense.sum(axis=0)
        assert len(col_sums) == 4
        assert col_sums[0] == pytest.approx(6.0)   # 1+5
        assert col_sums[1] == pytest.approx(3.0)   # 3
        assert col_sums[2] == pytest.approx(2.0)   # 2
        assert col_sums[3] == pytest.approx(10.0)  # 4+6
    
    def test_mean_total(self, small_csr_from_dense):
        """Test total mean of matrix."""
        # Mean: 21 / 12 = 1.75
        mean_val = small_csr_from_dense.mean()
        assert mean_val == pytest.approx(21.0 / 12.0)
    
    def test_mean_axis1_row_means(self, small_csr_from_dense):
        """Test row means (axis=1)."""
        row_means = small_csr_from_dense.mean(axis=1)
        assert len(row_means) == 3
        assert row_means[0] == pytest.approx(3.0 / 4.0)
        assert row_means[1] == pytest.approx(7.0 / 4.0)
        assert row_means[2] == pytest.approx(11.0 / 4.0)
    
    def test_min_total(self, small_csr_from_dense):
        """Test global minimum."""
        min_val = small_csr_from_dense.min()
        # Has implicit zeros
        assert min_val == pytest.approx(0.0)
    
    def test_max_total(self, small_csr_from_dense):
        """Test global maximum."""
        max_val = small_csr_from_dense.max()
        assert max_val == pytest.approx(6.0)
    
    def test_max_axis1_row_maxs(self, small_csr_from_dense):
        """Test row maximums (axis=1)."""
        row_maxs = small_csr_from_dense.max(axis=1)
        assert len(row_maxs) == 3
        assert row_maxs[0] == pytest.approx(2.0)
        assert row_maxs[1] == pytest.approx(4.0)
        assert row_maxs[2] == pytest.approx(6.0)


class TestCSCStatisticalMethods:
    """Test SclCSC statistical methods."""
    
    def test_sum_total(self, small_csc_from_dense):
        """Test total sum of CSC matrix."""
        total = small_csc_from_dense.sum()
        assert total == pytest.approx(21.0)
    
    def test_sum_axis0_col_sums(self, small_csc_from_dense):
        """Test column sums (axis=0) - efficient for CSC."""
        col_sums = small_csc_from_dense.sum(axis=0)
        assert len(col_sums) == 4
        assert col_sums[0] == pytest.approx(6.0)
        assert col_sums[1] == pytest.approx(3.0)
        assert col_sums[2] == pytest.approx(2.0)
        assert col_sums[3] == pytest.approx(10.0)
    
    def test_mean_total(self, small_csc_from_dense):
        """Test total mean of CSC matrix."""
        mean_val = small_csc_from_dense.mean()
        assert mean_val == pytest.approx(21.0 / 12.0)
    
    def test_max_total(self, small_csc_from_dense):
        """Test global maximum of CSC matrix."""
        max_val = small_csc_from_dense.max()
        assert max_val == pytest.approx(6.0)


# =============================================================================
# Conversion Alias Tests
# =============================================================================

class TestConversionAliases:
    """Test to_csc() and to_csr() aliases."""
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_to_csc_alias(self, small_csr_from_dense):
        """Test to_csc() is alias for tocsc()."""
        csc = small_csr_from_dense.to_csc()
        assert isinstance(csc, SclCSC)
        assert csc.shape == small_csr_from_dense.shape
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_to_csr_alias(self, small_csc_from_dense):
        """Test to_csr() is alias for tocsr()."""
        csr = small_csc_from_dense.to_csr()
        assert isinstance(csr, SclCSR)
        assert csr.shape == small_csc_from_dense.shape
    
    def test_to_csr_on_csr_returns_self(self, small_csr_from_dense):
        """Test to_csr() on CSR returns self."""
        result = small_csr_from_dense.to_csr()
        assert result is small_csr_from_dense
    
    def test_to_csc_on_csc_returns_self(self, small_csc_from_dense):
        """Test to_csc() on CSC returns self."""
        result = small_csc_from_dense.to_csc()
        assert result is small_csc_from_dense
