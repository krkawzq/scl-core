"""
Tests for sparse matrix operations.

Tests the operation functions in scl.sparse._ops:
- Stacking: vstack, hstack, concatenate
- Conversion: convert_format, from_scipy, to_scipy, etc.
- Alignment: align_rows, align_cols, align_to_categories
- Statistics: sum_rows, sum_cols, mean_rows, mean_cols, var_rows, var_cols
"""

import pytest
import numpy as np

try:
    from scl.sparse import (
        SclCSR, SclCSC, Array,
        Backend, Ownership,
        # Stacking
        vstack_csr, hstack_csc, vstack, hstack, concatenate,
        # Conversion
        convert_format, from_scipy, to_scipy,
        # Alignment
        align_rows, align_cols, align_to_categories,
        # Statistics
        sum_rows, sum_cols, mean_rows, mean_cols, var_rows, var_cols,
        # Utilities
        empty_like, zeros_like,
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
# Stacking Operations Tests
# =============================================================================

class TestStackingOperations:
    """Test stacking operations."""
    
    def test_vstack_csr_basic(self, requires_scl):
        """Test basic vstack_csr."""
        mat1 = SclCSR.from_dense([[1, 2], [3, 4]])
        mat2 = SclCSR.from_dense([[5, 6]])
        
        stacked = vstack_csr([mat1, mat2])
        
        assert stacked.shape == (3, 2)
        assert stacked.backend == Backend.VIRTUAL
    
    def test_vstack_csr_values(self, requires_scl):
        """Test vstack_csr preserves values."""
        mat1 = SclCSR.from_dense([[1, 2]])
        mat2 = SclCSR.from_dense([[3, 4]])
        
        stacked = vstack_csr([mat1, mat2]).to_owned()
        
        assert stacked[0, 0] == pytest.approx(1.0)
        assert stacked[0, 1] == pytest.approx(2.0)
        assert stacked[1, 0] == pytest.approx(3.0)
        assert stacked[1, 1] == pytest.approx(4.0)
    
    def test_vstack_csr_three_matrices(self, requires_scl):
        """Test vstack_csr with three matrices."""
        mat1 = SclCSR.from_dense([[1, 2]])
        mat2 = SclCSR.from_dense([[3, 4]])
        mat3 = SclCSR.from_dense([[5, 6]])
        
        stacked = vstack_csr([mat1, mat2, mat3])
        
        assert stacked.shape == (3, 2)
    
    def test_hstack_csc_basic(self, requires_scl):
        """Test basic hstack_csc."""
        mat1 = SclCSC.from_dense([[1, 2], [3, 4]])
        mat2 = SclCSC.from_dense([[5], [6]])
        
        stacked = hstack_csc([mat1, mat2])
        
        assert stacked.shape == (2, 3)
        assert stacked.backend == Backend.VIRTUAL
    
    def test_hstack_csc_values(self, requires_scl):
        """Test hstack_csc preserves values."""
        mat1 = SclCSC.from_dense([[1], [2]])
        mat2 = SclCSC.from_dense([[3], [4]])
        
        stacked = hstack_csc([mat1, mat2]).to_owned()
        
        assert stacked[0, 0] == pytest.approx(1.0)
        assert stacked[1, 0] == pytest.approx(2.0)
        assert stacked[0, 1] == pytest.approx(3.0)
        assert stacked[1, 1] == pytest.approx(4.0)
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_vstack_auto_conversion(self, requires_scl):
        """Test vstack auto-converts CSC to CSR."""
        csr = SclCSR.from_dense([[1, 2]])
        csc = SclCSC.from_dense([[3, 4]])
        
        stacked = vstack([csr, csc])
        assert stacked.shape == (2, 2)
    
    def test_concatenate_axis0(self, requires_scl):
        """Test concatenate with axis=0."""
        mat1 = SclCSR.from_dense([[1, 2]])
        mat2 = SclCSR.from_dense([[3, 4]])
        
        result = concatenate([mat1, mat2], axis=0)
        assert result.shape == (2, 2)


# =============================================================================
# Conversion Tests
# =============================================================================

class TestConversionOperations:
    """Test conversion operations."""
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_convert_format_csr_to_csc(self, requires_scl):
        """Test convert_format CSR to CSC."""
        csr = SclCSR.from_dense([[1, 2], [3, 4]])
        
        csc = convert_format(csr, 'csc')
        
        assert isinstance(csc, SclCSC)
        assert csc.shape == csr.shape
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_convert_format_csc_to_csr(self, requires_scl):
        """Test convert_format CSC to CSR."""
        csc = SclCSC.from_dense([[1, 2], [3, 4]])
        
        csr = convert_format(csc, 'csr')
        
        assert isinstance(csr, SclCSR)
        assert csr.shape == csc.shape
    
    def test_convert_format_same(self, requires_scl):
        """Test convert_format to same format returns same object."""
        csr = SclCSR.from_dense([[1, 2]])
        
        result = convert_format(csr, 'csr')
        assert result is csr
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_from_scipy_csr(self, requires_scl):
        """Test from_scipy with CSR matrix."""
        scipy_mat = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.float64)
        
        mat = from_scipy(scipy_mat)
        
        assert isinstance(mat, SclCSR)
        assert mat.shape == (2, 2)
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_from_scipy_csc(self, requires_scl):
        """Test from_scipy with CSC matrix."""
        scipy_mat = sp.csc_matrix([[1, 2], [3, 4]], dtype=np.float64)
        
        mat = from_scipy(scipy_mat)
        
        assert isinstance(mat, SclCSC)
        assert mat.shape == (2, 2)
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_to_scipy_csr(self, requires_scl):
        """Test to_scipy for CSR."""
        mat = SclCSR.from_dense([[1, 2], [3, 4]])
        
        scipy_mat = to_scipy(mat)
        
        assert isinstance(scipy_mat, sp.csr_matrix)
        assert scipy_mat.shape == (2, 2)
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_to_scipy_csc(self, requires_scl):
        """Test to_scipy for CSC."""
        mat = SclCSC.from_dense([[1, 2], [3, 4]])
        
        scipy_mat = to_scipy(mat)
        
        assert isinstance(scipy_mat, sp.csc_matrix)


# =============================================================================
# Alignment Tests
# =============================================================================

class TestAlignmentOperations:
    """Test alignment operations."""
    
    def test_align_rows_reorder(self, requires_scl):
        """Test align_rows reordering."""
        mat = SclCSR.from_dense([[1, 2], [3, 4], [5, 6]])
        
        # Reorder: [row2, row0, row1]
        aligned = align_rows(mat, [2, 0, 1], new_rows=3)
        
        assert aligned.shape == (3, 2)
        assert aligned[0, 0] == pytest.approx(5.0)  # was row 2
        assert aligned[1, 0] == pytest.approx(1.0)  # was row 0
        assert aligned[2, 0] == pytest.approx(3.0)  # was row 1
    
    def test_align_rows_subset(self, requires_scl):
        """Test align_rows subsetting."""
        mat = SclCSR.from_dense([[1, 2], [3, 4], [5, 6]])
        
        # Select rows 0 and 2 only
        aligned = align_rows(mat, [0, 2], new_rows=2)
        
        assert aligned.shape == (2, 2)
        assert aligned[0, 0] == pytest.approx(1.0)
        assert aligned[1, 0] == pytest.approx(5.0)
    
    def test_align_rows_with_empty(self, requires_scl):
        """Test align_rows with empty row (index -1)."""
        mat = SclCSR.from_dense([[1, 2], [3, 4]])
        
        # [row0, empty, row1]
        aligned = align_rows(mat, [0, -1, 1], new_rows=3)
        
        assert aligned.shape == (3, 2)
        assert aligned[0, 0] == pytest.approx(1.0)
        assert aligned[1, 0] == pytest.approx(0.0)  # Empty row
        assert aligned[2, 0] == pytest.approx(3.0)
    
    def test_align_cols_reorder(self, requires_scl):
        """Test align_cols reordering."""
        mat = SclCSC.from_dense([[1, 2, 3], [4, 5, 6]])
        
        # Reorder: [col2, col0]
        aligned = align_cols(mat, [2, 0], new_cols=2)
        
        assert aligned.shape == (2, 2)
        assert aligned[0, 0] == pytest.approx(3.0)  # was col 2
        assert aligned[0, 1] == pytest.approx(1.0)  # was col 0
    
    def test_align_to_categories_basic(self, requires_scl):
        """Test align_to_categories basic usage."""
        mat = SclCSR.from_dense([[1, 2, 3], [4, 5, 6]])
        
        source_cats = ['A', 'B', 'C']
        target_cats = ['C', 'A']  # Reorder and subset
        
        aligned = align_to_categories(mat, source_cats, target_cats, axis=1)
        
        assert aligned.shape == (2, 2)
    
    def test_align_to_categories_missing(self, requires_scl):
        """Test align_to_categories with missing categories."""
        mat = SclCSR.from_dense([[1, 2], [3, 4]])
        
        source_cats = ['A', 'B']
        target_cats = ['A', 'C', 'B']  # C is missing
        
        aligned = align_to_categories(mat, source_cats, target_cats, axis=1)
        
        assert aligned.shape == (2, 3)


# =============================================================================
# Statistics Tests
# =============================================================================

class TestStatisticsOperations:
    """Test statistics operations."""
    
    def test_sum_rows(self, requires_scl):
        """Test sum_rows."""
        mat = SclCSR.from_dense([[1, 2, 3], [4, 5, 6]])
        
        sums = sum_rows(mat)
        
        assert len(sums) == 2
        assert sums[0] == pytest.approx(6.0)  # 1+2+3
        assert sums[1] == pytest.approx(15.0)  # 4+5+6
    
    def test_sum_cols(self, requires_scl):
        """Test sum_cols."""
        mat = SclCSC.from_dense([[1, 2, 3], [4, 5, 6]])
        
        sums = sum_cols(mat)
        
        assert len(sums) == 3
        assert sums[0] == pytest.approx(5.0)  # 1+4
        assert sums[1] == pytest.approx(7.0)  # 2+5
        assert sums[2] == pytest.approx(9.0)  # 3+6
    
    def test_mean_rows(self, requires_scl):
        """Test mean_rows."""
        mat = SclCSR.from_dense([[3, 6, 9], [1, 2, 3]])
        
        means = mean_rows(mat)
        
        assert len(means) == 2
        assert means[0] == pytest.approx(6.0)  # (3+6+9)/3
        assert means[1] == pytest.approx(2.0)  # (1+2+3)/3
    
    def test_mean_cols(self, requires_scl):
        """Test mean_cols."""
        mat = SclCSC.from_dense([[2, 4], [4, 8]])
        
        means = mean_cols(mat)
        
        assert len(means) == 2
        assert means[0] == pytest.approx(3.0)  # (2+4)/2
        assert means[1] == pytest.approx(6.0)  # (4+8)/2
    
    def test_var_rows(self, requires_scl):
        """Test var_rows."""
        # Variance of [1,2,3] = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        mat = SclCSR.from_dense([[1, 2, 3]])
        
        vars_ = var_rows(mat)
        
        assert len(vars_) == 1
        expected = ((1-2)**2 + (2-2)**2 + (3-2)**2) / 3
        assert vars_[0] == pytest.approx(expected, rel=1e-5)
    
    def test_var_cols(self, requires_scl):
        """Test var_cols."""
        mat = SclCSC.from_dense([[1, 4], [3, 8]])
        
        vars_ = var_cols(mat)
        
        assert len(vars_) == 2
        # Col 0: [1, 3], mean=2, var=((1-2)^2 + (3-2)^2)/2 = 1
        # Col 1: [4, 8], mean=6, var=((4-6)^2 + (8-6)^2)/2 = 4
        assert vars_[0] == pytest.approx(1.0, rel=1e-5)
        assert vars_[1] == pytest.approx(4.0, rel=1e-5)
    
    def test_sum_rows_with_zeros(self, requires_scl):
        """Test sum_rows handles zero elements correctly."""
        mat = SclCSR.from_dense([[1, 0, 2], [0, 0, 0]])
        
        sums = sum_rows(mat)
        
        assert sums[0] == pytest.approx(3.0)
        assert sums[1] == pytest.approx(0.0)


# =============================================================================
# Utility Functions Tests
# =============================================================================

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_empty_like(self, requires_scl):
        """Test empty_like function."""
        mat = SclCSR.from_dense([[1, 2], [3, 4]])
        
        empty = empty_like(mat, nnz=10)
        
        assert empty.shape == mat.shape
        assert empty.dtype == mat.dtype
        assert empty.nnz == 10
    
    def test_zeros_like(self, requires_scl):
        """Test zeros_like function."""
        mat = SclCSR.from_dense([[1, 2], [3, 4]])
        
        zero = zeros_like(mat)
        
        assert zero.shape == mat.shape
        assert zero.dtype == mat.dtype
        assert zero.nnz == 0


# =============================================================================
# Preprocessing Operations Tests
# =============================================================================

class TestPreprocessingOperations:
    """Test preprocessing operations."""
    
    def test_normalize_l1(self, requires_scl):
        """Test L1 normalization."""
        import scl.preprocessing as pp
        
        mat = SclCSR.from_dense([[1, 2, 3], [4, 5, 6]])
        normalized = pp.normalize(mat, norm='l1', axis=1)
        
        # Each row should sum to 1
        row_sums = normalized.sum(axis=1)
        assert row_sums[0] == pytest.approx(1.0)
        assert row_sums[1] == pytest.approx(1.0)
    
    def test_normalize_l2(self, requires_scl):
        """Test L2 normalization."""
        import scl.preprocessing as pp
        
        mat = SclCSR.from_dense([[3, 4], [5, 12]])
        normalized = pp.normalize(mat, norm='l2', axis=1)
        
        # Each row should have unit norm
        row0 = normalized[0, :].to_dense()
        row1 = normalized[1, :].to_dense()
        
        norm0 = np.sqrt(row0[0]**2 + row0[1]**2)
        norm1 = np.sqrt(row1[0]**2 + row1[1]**2)
        
        assert norm0 == pytest.approx(1.0)
        assert norm1 == pytest.approx(1.0)
    
    def test_log1p_transform(self, requires_scl):
        """Test log1p transformation."""
        import scl.preprocessing as pp
        
        mat = SclCSR.from_dense([[1, 2], [3, 4]])
        transformed = pp.log1p(mat)
        
        # log1p should reduce values
        assert transformed.sum() < mat.sum()
        # Shape preserved
        assert transformed.shape == mat.shape
    
    def test_scale_operation(self, requires_scl):
        """Test scale operation."""
        import scl.preprocessing as pp
        
        mat = SclCSR.from_dense([[1, 2], [3, 4]])
        
        # Scale rows by [2, 0.5]
        row_factors = Array.from_list([2.0, 0.5], dtype='float64')
        scaled = pp.scale(mat, row_factors=row_factors)
        
        # Row 0 doubled, row 1 halved
        assert scaled[0, 0] == pytest.approx(2.0)
        assert scaled[1, 0] == pytest.approx(1.5)


# =============================================================================
# Statistical Tests
# =============================================================================

class TestStatisticalTests:
    """Test statistical test operations."""
    
    def test_mwu_test_basic(self, requires_scl):
        """Test Mann-Whitney U test."""
        import scl.statistics as stats
        
        # Create test matrix with two groups
        mat = SclCSC.from_dense([
            [1, 2, 3],  # Group 0
            [1, 2, 3],  # Group 0
            [4, 5, 6],  # Group 1
            [4, 5, 6]   # Group 1
        ])
        
        groups = Array.from_list([0, 0, 1, 1], dtype='int32')
        
        u_stats, p_values, log2_fc = stats.mwu_test(mat, groups)
        
        assert len(u_stats) == 3
        assert len(p_values) == 3
        assert len(log2_fc) == 3
        # All genes should show difference between groups
        assert all(p < 0.5 for p in p_values)
    
    def test_ttest_basic(self, requires_scl):
        """Test T-test."""
        import scl.statistics as stats
        
        # Create test matrix
        mat = SclCSC.from_dense([
            [1, 2],  # Group 0
            [1, 2],  # Group 0
            [5, 6],  # Group 1
            [5, 6]   # Group 1
        ])
        
        groups = Array.from_list([0, 0, 1, 1], dtype='int32')
        
        t_stats, p_values, log2_fc = stats.ttest(mat, groups)
        
        assert len(t_stats) == 2
        assert len(p_values) == 2
        assert len(log2_fc) == 2

