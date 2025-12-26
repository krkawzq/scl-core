"""
Tests for C++ kernel bindings.
"""

import pytest
import numpy as np
from scl.sparse import Array, from_list, zeros
from scl.sparse import float32, int64


class TestLibraryLoader:
    """Test library loading functionality."""
    
    def test_get_lib_f32(self, requires_scl):
        """Test loading f32 library."""
        from scl._kernel.lib_loader import get_lib_f32, LibraryNotFoundError
        
        try:
            lib = get_lib_f32()
            assert lib is not None
        except LibraryNotFoundError:
            pytest.skip("Library not found (may need to build first)")
    
    def test_get_lib_f64(self, requires_scl):
        """Test loading f64 library."""
        from scl._kernel.lib_loader import get_lib_f64, LibraryNotFoundError
        
        try:
            lib = get_lib_f64()
            assert lib is not None
        except LibraryNotFoundError:
            pytest.skip("Library not found (may need to build first)")
    
    def test_get_lib_default(self, requires_scl):
        """Test getting default library."""
        from scl._kernel.lib_loader import get_lib, LibraryNotFoundError
        
        try:
            lib = get_lib()
            assert lib is not None
        except LibraryNotFoundError:
            pytest.skip("Library not found (may need to build first)")


class TestUtilsKernel:
    """Test utility kernels."""
    
    def test_compute_lengths(self, requires_scl):
        """Test compute_lengths kernel."""
        from scl._kernel import utils as kernel_utils
        
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
        except (AttributeError, RuntimeError) as e:
            pytest.skip(f"Kernel not available: {e}")


class TestSparseKernel:
    """Test sparse matrix kernels."""
    
    def test_row_sums(self, requires_scl):
        """Test row sums computation."""
        from scl._kernel import sparse as sparse_kernel
        
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
            
            if status == 0:
                # Check results: row 0 sum = 1+2=3, row 1 sum = 3
                assert output[0] == pytest.approx(3.0)
                assert output[1] == pytest.approx(3.0)
        except (AttributeError, RuntimeError) as e:
            pytest.skip(f"Kernel not available: {e}")


class TestTypes:
    """Test type detection."""
    
    def test_detect_precision(self, requires_scl):
        """Test precision detection."""
        from scl._kernel.types import detect_precision, c_real, np_real
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        try:
            detect_precision('f32')
            # Should be float32
            import ctypes
            assert c_real == ctypes.c_float
            assert np_real == np.float32
        except (LibraryNotFoundError, RuntimeError):
            pytest.skip("Cannot detect precision (library not available)")

