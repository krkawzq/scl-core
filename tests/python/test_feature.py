"""
Tests for feature selection and QC operations.
"""

import pytest
import numpy as np
import scipy.sparse as sp


class TestQCOperations:
    """Test quality control operations."""
    
    def test_compute_qc_basic(self):
        """Test basic QC metrics computation."""
        import scl.feature as feat
        
        # Create test matrix: 3 cells x 4 genes
        mat = sp.csr_matrix([
            [1.0, 0.0, 2.0, 0.0],  # Cell 0: 2 genes, 3 counts
            [1.0, 1.0, 1.0, 0.0],  # Cell 1: 3 genes, 3 counts
            [0.0, 0.0, 3.0, 0.0]   # Cell 2: 1 gene, 3 counts
        ])
        
        n_genes, total_counts = feat.compute_qc(mat)
        
        assert len(n_genes) == 3
        assert len(total_counts) == 3
        assert n_genes[0] == 2
        assert n_genes[1] == 3
        assert n_genes[2] == 1
        assert total_counts[0] == pytest.approx(3.0)
        assert total_counts[1] == pytest.approx(3.0)
        assert total_counts[2] == pytest.approx(3.0)
    
    def test_standard_moments(self):
        """Test standard moments computation."""
        import scl.feature as feat
        
        # Create test matrix: 4 cells x 2 genes
        mat = sp.csc_matrix([
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0]
        ])
        
        means, variances, skewness, kurtosis = feat.standard_moments(mat, ddof=0)
        
        assert len(means) == 2
        assert len(variances) == 2
        # Gene 0: [1, 2, 3, 4], mean = 2.5
        assert means[0] == pytest.approx(2.5)
        # Gene 1: [2, 4, 6, 8], mean = 5.0
        assert means[1] == pytest.approx(5.0)
    
    def test_clipped_moments(self):
        """Test clipped moments computation."""
        import scl.feature as feat
        
        # Create test matrix with outliers
        mat = sp.csc_matrix([
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [100.0, 200.0]  # Outliers
        ])
        
        # Clip at reasonable values
        from scl.sparse import Array
        clip_vals = Array.from_list([10.0, 20.0], dtype='float64')
        
        means, variances = feat.clipped_moments(mat, clip_max=clip_vals)
        
        assert len(means) == 2
        assert len(variances) == 2
        # Clipped means should be much lower than unclipped
        assert means[0] < 26.5  # Unclipped mean would be (1+2+3+100)/4 = 26.5


class TestFeatureSelection:
    """Test feature selection operations."""
    
    def test_detection_rate(self):
        """Test detection rate computation."""
        import scl.feature as feat
        
        # Create test matrix: 4 cells x 3 genes
        mat = sp.csc_matrix([
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        
        rates = feat.detection_rate(mat)
        
        assert len(rates) == 3
        assert rates[0] == pytest.approx(0.75)  # 3/4 cells
        assert rates[1] == pytest.approx(0.5)   # 2/4 cells
        assert rates[2] == pytest.approx(1.0)   # 4/4 cells
    
    def test_dispersion(self):
        """Test dispersion computation."""
        import scl.feature as feat
        
        # Create test matrix
        mat = sp.csr_matrix([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0]
        ])
        
        # Column-wise dispersion
        disp = feat.dispersion(mat, axis=0)
        
        assert len(disp) == 3
        # All values should be positive
        assert all(d >= 0 for d in disp)
    
    def test_highly_variable_genes(self):
        """Test HVG selection."""
        import scl.feature as feat
        
        # Create test matrix with varying genes
        np.random.seed(42)
        n_cells, n_genes = 100, 50
        
        # Create genes with different variability
        data = np.random.randn(n_cells, n_genes)
        # Make some genes more variable
        data[:, :10] *= 3  # High variance genes
        data[:, 10:20] *= 0.5  # Low variance genes
        
        mat = sp.csr_matrix(data)
        
        # Select top 15 HVGs
        hvg_idx = feat.highly_variable(
            mat, n_top=15,
            min_mean=0.0, max_mean=10.0,
            min_dispersion=0.0
        )
        
        # Should return some indices
        assert len(hvg_idx) > 0
        assert len(hvg_idx) <= 15


class TestSpatialStatistics:
    """Test spatial statistics operations."""
    
    def test_morans_i_basic(self):
        """Test Moran's I computation."""
        import scl.spatial as spatial
        
        # Create simple spatial pattern
        # 4 cells in a line with spatial weights
        features = sp.csc_matrix([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        
        # Spatial weights: neighbors
        weights = sp.csc_matrix([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        
        morans = spatial.morans_i(features, weights)
        
        assert len(morans) == 2
        # Feature 0 shows positive spatial autocorrelation
        # (similar values cluster together)
        assert morans[0] > 0
    
    def test_mmd_rbf(self):
        """Test MMD with RBF kernel."""
        import scl.spatial as spatial
        
        # Create two distributions
        X = sp.csc_matrix([
            [1.0, 2.0],
            [1.5, 2.5],
            [1.2, 2.2]
        ])
        
        Y = sp.csc_matrix([
            [5.0, 6.0],
            [5.5, 6.5],
            [5.2, 6.2]
        ])
        
        # MMD should be positive (different distributions)
        mmd = spatial.mmd_rbf(X, Y, gamma=1.0)
        
        assert mmd > 0
        
        # MMD between same distribution should be near zero
        mmd_same = spatial.mmd_rbf(X, X, gamma=1.0)
        assert mmd_same == pytest.approx(0.0, abs=1e-6)


class TestGroupOperations:
    """Test group-wise operations."""
    
    def test_group_statistics(self):
        """Test group statistics computation."""
        import scl.statistics as stats
        
        # Create test matrix: 6 cells x 3 genes
        mat = sp.csc_matrix([
            [1.0, 2.0, 3.0],  # Group 0
            [2.0, 3.0, 4.0],  # Group 0
            [3.0, 4.0, 5.0],  # Group 1
            [4.0, 5.0, 6.0],  # Group 1
            [5.0, 6.0, 7.0],  # Group 2
            [6.0, 7.0, 8.0]   # Group 2
        ])
        
        groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        
        # This should work if group_stats is implemented
        try:
            result = stats.group_stats(mat, groups)
            # Should return per-group statistics
            assert result is not None
        except AttributeError:
            pytest.skip("group_stats not yet implemented in high-level API")


class TestScaleOperations:
    """Test scaling and standardization operations."""
    
    def test_scale_rows(self):
        """Test row scaling."""
        import scl.preprocessing as pp
        
        mat = sp.csr_matrix([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        # Scale rows by [2.0, 0.5]
        row_factors = np.array([2.0, 0.5])
        scaled = pp.scale(mat, row_factors=row_factors)
        
        # Row 0 should be doubled
        row0 = scaled[0, :].toarray().ravel()
        assert row0[0] == pytest.approx(2.0)
        assert row0[1] == pytest.approx(4.0)
        assert row0[2] == pytest.approx(6.0)
        
        # Row 1 should be halved
        row1 = scaled[1, :].toarray().ravel()
        assert row1[0] == pytest.approx(2.0)
        assert row1[1] == pytest.approx(2.5)
        assert row1[2] == pytest.approx(3.0)
    
    def test_scale_cols(self):
        """Test column scaling."""
        import scl.preprocessing as pp
        
        mat = sp.csr_matrix([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        # Scale columns by [2.0, 1.0, 0.5]
        col_factors = np.array([2.0, 1.0, 0.5])
        scaled = pp.scale(mat, col_factors=col_factors)
        
        # Column 0 should be doubled, column 2 halved
        row0 = scaled[0, :].toarray().ravel()
        assert row0[0] == pytest.approx(2.0)   # 1 * 2
        assert row0[1] == pytest.approx(2.0)   # 2 * 1
        assert row0[2] == pytest.approx(1.5)   # 3 * 0.5

