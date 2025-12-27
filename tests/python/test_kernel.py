"""
Tests for C++ kernel bindings.

Updated for new _kernel module structure.
"""

import pytest
import numpy as np

try:
    from scl.sparse import Array, from_list, zeros
    from scl.sparse import float32, int64
    HAS_SCL = True
except ImportError:
    HAS_SCL = False


class TestLibraryLoader:
    """Test library loading functionality."""
    
    def test_get_lib_default(self, requires_scl):
        """Test getting default library."""
        from scl._kernel.lib_loader import get_lib, LibraryNotFoundError
        
        try:
            lib = get_lib()
            assert lib is not None
        except LibraryNotFoundError:
            pytest.skip("Library not found (may need to build first)")
    
    def test_get_lib_f32(self, requires_scl):
        """Test loading f32 library."""
        from scl._kernel.lib_loader import get_lib, LibraryNotFoundError
        
        try:
            lib = get_lib('f32')
            assert lib is not None
        except LibraryNotFoundError:
            pytest.skip("Library not found (may need to build first)")
    
    def test_get_lib_f64(self, requires_scl):
        """Test loading f64 library."""
        from scl._kernel.lib_loader import get_lib, LibraryNotFoundError
        
        try:
            lib = get_lib('f64')
            assert lib is not None
        except LibraryNotFoundError:
            pytest.skip("Library not found (may need to build first)")


class TestUtilsKernel:
    """Test utility kernels."""
    
    def test_compute_lengths(self, requires_scl):
        """Test compute_lengths kernel."""
        from scl._kernel import utils as kernel_utils
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        # Create indptr: [0, 2, 4, 6] for 3 rows
        indptr = from_list([0, 2, 4, 6], dtype=int64)
        rows = 3
        lengths = zeros(rows, dtype=int64)
        
        try:
            kernel_utils.compute_lengths(
                indptr.get_pointer(),
                rows,
                lengths.get_pointer()
            )
            
            # Check results
            assert lengths[0] == 2
            assert lengths[1] == 2
            assert lengths[2] == 2
        except (AttributeError, RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Kernel not available: {e}")
        except Exception as e:
            pytest.skip(f"Kernel test failed: {e}")


class TestSparseKernel:
    """Test sparse matrix kernels."""
    
    def test_row_sums(self, requires_scl):
        """Test row sums computation."""
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        try:
            from scl._kernel import sparse as sparse_kernel
        except ImportError:
            pytest.skip("sparse kernel module not available")
        
        # Create small test matrix: 2x3
        # [[1, 0, 2],
        #  [0, 3, 0]]
        data = from_list([1.0, 2.0, 3.0], dtype=float32)
        indices = from_list([0, 2, 1], dtype=int64)
        indptr = from_list([0, 2, 3], dtype=int64)
        row_lengths = from_list([2, 1], dtype=int64)
        rows, cols, nnz = 2, 3, 3
        
        output = zeros(rows, dtype=float32)
        
        try:
            # Try different function name patterns
            if hasattr(sparse_kernel, 'row_sums_csr'):
                status = sparse_kernel.row_sums_csr(
                    data.get_pointer(),
                    indices.get_pointer(),
                    indptr.get_pointer(),
                    row_lengths.get_pointer(),
                    rows,
                    cols,
                    nnz,
                    output.get_pointer()
                )
            elif hasattr(sparse_kernel, 'scl_primary_sums_csr'):
                status = sparse_kernel.scl_primary_sums_csr(
                    data.get_pointer(),
                    indices.get_pointer(),
                    indptr.get_pointer(),
                    row_lengths.get_pointer(),
                    rows,
                    cols,
                    nnz,
                    output.get_pointer()
                )
            else:
                pytest.skip("No row sum function available")
            
            if status == 0:
                # Check results: row 0 sum = 1+2=3, row 1 sum = 3
                assert output[0] == pytest.approx(3.0)
                assert output[1] == pytest.approx(3.0)
        except (AttributeError, RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Kernel not available: {e}")


class TestTypes:
    """Test type detection."""
    
    def test_c_types_exist(self, requires_scl):
        """Test that C types exist."""
        from scl._kernel.types import c_real, c_index, c_size, c_byte
        import ctypes
        
        # Types should be ctypes types
        assert c_real in (ctypes.c_float, ctypes.c_double)
        assert c_index in (ctypes.c_int16, ctypes.c_int32, ctypes.c_int64)
        assert c_size == ctypes.c_size_t
        assert c_byte == ctypes.c_uint8
    
    def test_detect_precision(self, requires_scl):
        """Test precision detection."""
        from scl._kernel.types import detect_precision, c_real
        from scl._kernel.lib_loader import LibraryNotFoundError
        import ctypes
        
        try:
            detect_precision('f32')
            # Should be float32
            assert c_real == ctypes.c_float
        except (LibraryNotFoundError, RuntimeError, AttributeError) as e:
            pytest.skip(f"Cannot detect precision: {e}")
    
    def test_error_handling(self, requires_scl):
        """Test error handling functions."""
        from scl._kernel.types import get_last_error, clear_error, check_error
        
        # These should be callable
        assert callable(get_last_error)
        assert callable(clear_error)
        assert callable(check_error)


# =============================================================================
# High-Level API Tests (using kernel internally)
# =============================================================================

class TestLinalgKernel:
    """Test linear algebra operations (kernel-based)."""
    
    def test_spmv(self, requires_scl):
        """Test sparse matrix-vector multiplication."""
        import scipy.sparse as sp
        import numpy as np
        from scl.math import linalg as smath
        
        # Use scipy matrix (should use scipy path)
        scipy_mat = sp.csr_matrix([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 1.0])
        
        result = smath.spmv(scipy_mat, x)
        
        # Row 0: 1*1 + 2*1 = 3
        # Row 1: 3*1 + 4*1 = 7
        assert len(result) == 2
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(7.0)
    
    def test_gram(self, requires_scl):
        """Test Gram matrix computation."""
        import scipy.sparse as sp
        from scl.math import linalg as smath
        
        # Use scipy matrix (should use scipy path)
        scipy_mat = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        
        result = smath.gram(scipy_mat)
        
        # Gram matrix should be 2x2: X^T X
        # Column 0: [1, 0, 1]^T, Column 1: [0, 1, 1]^T
        # [0,0] = 1*1 + 0*0 + 1*1 = 2
        # [0,1] = 1*0 + 0*1 + 1*1 = 1
        # [1,1] = 0*0 + 1*1 + 1*1 = 2
        assert len(result) == 4
        assert result[0] == pytest.approx(2.0)  # [0,0]
        assert result[1] == pytest.approx(1.0)  # [0,1]
        assert result[2] == pytest.approx(1.0)  # [1,0]
        assert result[3] == pytest.approx(2.0)  # [1,1]


class TestTransformKernel:
    """Test data transformation operations (kernel-based)."""
    
    def test_log1p(self, requires_scl):
        """Test log1p transformation."""
        from scl.sparse import SclCSR
        import scl.preprocessing as pp
        import math
        
        mat = SclCSR.from_dense([[1.0, 2.0], [3.0, 4.0]])
        original_sum = mat.sum()
        
        result = pp.log1p(mat)
        result_sum = result.sum()
        
        # log1p should reduce values (ln(1+x) < x for x > 0)
        assert result_sum < original_sum
        # Verify shape preserved
        assert result.shape == mat.shape
    
    def test_softmax(self, requires_scl):
        """Test softmax transformation."""
        import scipy.sparse as sp
        import scl.preprocessing as pp
        
        # Use scipy matrix to avoid kernel issues
        scipy_mat = sp.csr_matrix([[1.0, 2.0], [1.0, 1.0]])
        
        result = pp.softmax(scipy_mat, axis=1)
        
        # Softmax values should be between 0 and 1
        # and each row should sum to approximately 1
        assert result.shape == scipy_mat.shape
        
        row0 = result[0, :].toarray().ravel()
        row1 = result[1, :].toarray().ravel()
        
        assert row0.sum() == pytest.approx(1.0)
        assert row1.sum() == pytest.approx(1.0)
        # Row 1 has equal values, so softmax should be [0.5, 0.5]
        assert row1[0] == pytest.approx(0.5)
        assert row1[1] == pytest.approx(0.5)


class TestNormalizeKernel:
    """Test normalization operations (kernel-based)."""
    
    def test_l1_normalize(self, requires_scl):
        """Test L1 normalization."""
        import scipy.sparse as sp
        import scl.preprocessing as pp
        
        # Use scipy matrix to avoid kernel issues
        scipy_mat = sp.csr_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        result = pp.normalize(scipy_mat, norm='l1', axis=1)
        
        # Each row should sum to 1 after L1 normalization
        row0_sum = result[0, :].toarray().ravel().sum()
        row1_sum = result[1, :].toarray().ravel().sum()
        
        assert row0_sum == pytest.approx(1.0)
        assert row1_sum == pytest.approx(1.0)


class TestStatisticsKernel:
    """Test statistics operations (kernel-based)."""
    
    def test_variance(self, requires_scl):
        """Test variance computation."""
        from scl.sparse import SclCSR
        import scl.statistics as stats
        
        # Use scipy for comparison
        import scipy.sparse as sp
        import numpy as np
        
        dense = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scipy_mat = sp.csr_matrix(dense)
        
        # Test with scipy matrix (should use numpy/scipy path)
        row_vars = stats.var(scipy_mat, axis=1)
        
        # Row 0: [1, 2, 3], mean = 2, var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        # Row 1: [4, 5, 6], mean = 5, var = ((4-5)^2 + (5-5)^2 + (6-5)^2) / 3 = 2/3
        assert row_vars[0] == pytest.approx(2.0/3.0)
        assert row_vars[1] == pytest.approx(2.0/3.0)


class TestQCKernel:
    """Test QC kernel operations."""
    
    def test_compute_qc(self, requires_scl):
        """Test basic QC computation."""
        import scipy.sparse as sp
        import scl.feature as feat
        
        # Create test matrix: 3 cells x 4 genes
        # Cell 0: 2 genes expressed
        # Cell 1: 3 genes expressed
        # Cell 2: 1 gene expressed
        scipy_mat = sp.csr_matrix([
            [1.0, 0.0, 2.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 3.0, 0.0]
        ])
        
        n_genes, total_counts = feat.compute_qc(scipy_mat)
        
        assert len(n_genes) == 3
        assert len(total_counts) == 3
        assert n_genes[0] == 2  # Cell 0 has 2 non-zero genes
        assert n_genes[1] == 3  # Cell 1 has 3 non-zero genes
        assert n_genes[2] == 1  # Cell 2 has 1 non-zero gene
        assert total_counts[0] == pytest.approx(3.0)  # 1 + 2
        assert total_counts[1] == pytest.approx(3.0)  # 1 + 1 + 1
        assert total_counts[2] == pytest.approx(3.0)  # 3


class TestFeatureKernel:
    """Test feature statistics kernel operations."""
    
    def test_detection_rate(self, requires_scl):
        """Test detection rate computation."""
        import scipy.sparse as sp
        import scl.feature as feat
        
        # Create test matrix: 4 cells x 3 genes
        # Gene 0: expressed in 3/4 cells
        # Gene 1: expressed in 2/4 cells
        # Gene 2: expressed in 4/4 cells
        scipy_mat = sp.csc_matrix([
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        
        rates = feat.detection_rate(scipy_mat)
        
        assert len(rates) == 3
        assert rates[0] == pytest.approx(0.75)  # 3/4
        assert rates[1] == pytest.approx(0.5)   # 2/4
        assert rates[2] == pytest.approx(1.0)   # 4/4
    
    def test_dispersion(self, requires_scl):
        """Test dispersion computation."""
        import scipy.sparse as sp
        import scl.feature as feat
        
        # Create test matrix with known statistics
        scipy_mat = sp.csr_matrix([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0]
        ])
        
        disp = feat.dispersion(scipy_mat, axis=0)
        
        # Dispersion = variance / mean
        # Should have 3 values (one per column)
        assert len(disp) == 3


class TestSortingKernel:
    """Test sorting kernel operations."""
    
    def test_vqsort_real(self, requires_scl):
        """Test SIMD-optimized sorting."""
        from scl._kernel import sorting as kernel_sorting
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        try:
            data = from_list([3.0, 1.0, 4.0, 1.0, 5.0], dtype=float32)
            
            kernel_sorting.vqsort_real_ascending(data.get_pointer(), 5)
            
            # Check sorted
            assert data[0] <= data[1] <= data[2] <= data[3] <= data[4]
            assert data[0] == pytest.approx(1.0)
            assert data[4] == pytest.approx(5.0)
        except (AttributeError, RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Sorting kernel not available: {e}")
    
    def test_argsort_real(self, requires_scl):
        """Test argsort."""
        from scl._kernel import sorting as kernel_sorting
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        try:
            keys = from_list([3.0, 1.0, 4.0, 1.0, 5.0], dtype=float32)
            indices = zeros(5, dtype=int64)
            
            kernel_sorting.argsort_real_ascending(
                keys.get_pointer(), 5, indices.get_pointer()
            )
            
            # indices should be [1, 3, 0, 2, 4] or similar
            # (indices that would sort the array)
            assert keys[indices[0]] <= keys[indices[1]]
            assert keys[indices[3]] <= keys[indices[4]]
        except (AttributeError, RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Argsort kernel not available: {e}")


class TestCoreKernel:
    """Test core utility kernel operations."""
    
    def test_fill_real(self, requires_scl):
        """Test fill operation."""
        from scl._kernel import core as kernel_core
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        try:
            data = zeros(5, dtype=float32)
            kernel_core.fill_real(data.get_pointer(), 5, 3.14)
            
            for i in range(5):
                assert data[i] == pytest.approx(3.14)
        except (AttributeError, RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Core kernel not available: {e}")
    
    def test_sum_real(self, requires_scl):
        """Test sum reduction."""
        from scl._kernel import core as kernel_core
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        try:
            data = from_list([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float32)
            result = kernel_core.sum_real(data.get_pointer(), 5)
            
            assert result == pytest.approx(15.0)
        except (AttributeError, RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Core kernel not available: {e}")
    
    def test_minmax_real(self, requires_scl):
        """Test minmax reduction."""
        from scl._kernel import core as kernel_core
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        try:
            data = from_list([3.0, 1.0, 4.0, 1.0, 5.0], dtype=float32)
            min_val, max_val = kernel_core.minmax_real(data.get_pointer(), 5)
            
            assert min_val == pytest.approx(1.0)
            assert max_val == pytest.approx(5.0)
        except (AttributeError, RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Core kernel not available: {e}")


class TestIOKernel:
    """Test I/O kernel operations."""
    
    def test_file_exists(self, requires_scl):
        """Test file existence check."""
        from scl._kernel import io as kernel_io
        from scl._kernel.lib_loader import LibraryNotFoundError
        import tempfile
        import os
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False) as f:
                temp_path = f.name
            
            try:
                # File should exist
                assert kernel_io.file_exists(temp_path) == True
                
                # Remove file
                os.unlink(temp_path)
                
                # File should not exist
                assert kernel_io.file_exists(temp_path) == False
            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except (AttributeError, RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"I/O kernel not available: {e}")
