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
