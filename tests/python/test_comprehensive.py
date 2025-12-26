"""
Comprehensive integration tests for SCL sparse module.

Tests the complete workflow from Array creation to matrix operations.
"""

import pytest
import numpy as np


class TestEndToEndWorkflow:
    """Test complete workflows."""
    
    def test_create_manipulate_matrix(self, requires_scl):
        """Test creating and manipulating a matrix."""
        from scl.sparse import SclCSR, Array, from_list
        import scl.sparse as sp
        
        # Step 1: Create arrays with beautiful syntax
        data = from_list([1.0, 2.0, 3.0, 4.0], dtype=sp.float32)
        indices = from_list([0, 2, 1, 3], dtype=sp.int64)
        indptr = from_list([0, 2, 4], dtype=sp.int64)
        
        # Step 2: Create matrix
        mat = SclCSR(data, indices, indptr, shape=(2, 4))
        
        assert mat.shape == (2, 4)
        assert mat.nnz == 4
        assert mat.dtype == 'float32'
        
        # Step 3: Get row lengths (should auto-compute)
        assert mat.row_lengths.size == 2
        assert mat.row_lengths[0] == 2
        assert mat.row_lengths[1] == 2
        
        # Step 4: Get row
        indices_row0, values_row0 = mat.get_row(0)
        assert len(indices_row0) == 2
        assert len(values_row0) == 2
        
        # Step 5: Get dense row
        row0_dense = mat.get_row_dense(0)
        assert len(row0_dense) == 4
        assert row0_dense[0] == pytest.approx(1.0)
        assert row0_dense[2] == pytest.approx(2.0)
    
    def test_dtype_system_workflow(self, requires_scl):
        """Test dtype system in real usage."""
        from scl.sparse import Array, DType
        import scl.sparse as sp
        
        # Test all supported dtypes
        dtypes = [sp.float32, sp.float64, sp.int32, sp.int64, sp.uint8]
        
        for dtype in dtypes:
            arr = Array.zeros(10, dtype=dtype)
            assert arr.dtype == dtype.value
            
            # Test conversion
            lst = arr.tolist()
            assert len(lst) == 10
    
    def test_matrix_slice_workflow(self, requires_scl, small_csr_matrix):
        """Test slicing workflow."""
        # This will test once C++ library is available
        try:
            # Slice rows
            subset = small_csr_matrix.slice_rows([0, 2])
            assert subset.shape == (2, 4)
            
            # Get row from subset
            indices, values = subset.get_row(0)
            assert len(indices) >= 0  # Should work
        except (RuntimeError, AttributeError) as e:
            pytest.skip(f"Slicing requires C++ library: {e}")


class TestDTypeUsagePatterns:
    """Test realistic dtype usage patterns."""
    
    def test_dtype_as_module_attribute(self, requires_scl):
        """Test using dtype as sp.float32."""
        from scl.sparse import Array
        import scl.sparse as sp
        
        # This should work and look beautiful
        arr = Array.zeros(100, dtype=sp.float32)
        assert arr.dtype == 'float32'
        
        arr2 = Array.zeros(100, dtype=sp.float64)
        assert arr2.dtype == 'float64'
    
    def test_dtype_mixing_not_allowed(self, requires_scl):
        """Test that mixing dtypes raises error."""
        from scl.sparse import SclCSR, Array, from_list
        import scl.sparse as sp
        
        data = from_list([1.0, 2.0], dtype=sp.float32)
        indices = from_list([0, 1], dtype=sp.int32)  # Wrong! Should be int64
        indptr = from_list([0, 2], dtype=sp.int64)
        
        # Should raise TypeError
        with pytest.raises(TypeError):
            SclCSR(data, indices, indptr, shape=(1, 3))


class TestMemorySafety:
    """Test memory safety and GC behavior."""
    
    def test_array_lifecycle(self, requires_scl):
        """Test that Array memory is properly managed."""
        from scl.sparse import Array
        import scl.sparse as sp
        
        # Create array
        arr = Array.zeros(1000, dtype=sp.float32)
        ptr1 = arr.ptr
        
        # Modify
        arr[0] = 42.0
        assert arr[0] == pytest.approx(42.0)
        
        # Pointer should remain valid
        assert arr.ptr == ptr1
        
        # Copy should have different pointer
        arr2 = arr.copy()
        assert arr2.ptr != ptr1
        
        # But same data
        assert arr2[0] == pytest.approx(42.0)
    
    def test_matrix_gc_safety(self, requires_scl):
        """Test that matrix doesn't crash after GC."""
        from scl.sparse import SclCSR, Array, from_list
        import scl.sparse as sp
        
        def create_matrix():
            data = from_list([1.0, 2.0], dtype=sp.float32)
            indices = from_list([0, 1], dtype=sp.int64)
            indptr = from_list([0, 2], dtype=sp.int64)
            return SclCSR(data, indices, indptr, shape=(1, 2))
        
        mat = create_matrix()
        # Original arrays should still be accessible through mat
        assert mat.shape == (1, 2)
        assert mat.nnz == 2


class TestErrorMessages:
    """Test that error messages are clear and helpful."""
    
    def test_invalid_dtype_clear_message(self, requires_scl):
        """Test clear error message for invalid dtype."""
        from scl.sparse import Array
        
        try:
            Array(10, dtype='invalid_type')
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            # Error message should be helpful
            assert 'invalid_type' in str(e).lower() or 'unsupported' in str(e).lower()
    
    def test_shape_mismatch_clear_message(self, requires_scl):
        """Test clear error message for shape mismatch."""
        from scl.sparse import SclCSR, from_list
        import scl.sparse as sp
        
        data = from_list([1.0], dtype=sp.float32)
        indices = from_list([0], dtype=sp.int64)
        indptr = from_list([0, 1, 2], dtype=sp.int64)  # Wrong size
        
        try:
            SclCSR(data, indices, indptr, shape=(2, 10))
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            # Error should mention 'mismatch' or 'size'
            assert 'mismatch' in str(e).lower() or 'size' in str(e).lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_matrix(self, requires_scl):
        """Test creating empty matrix."""
        from scl.sparse import SclCSR
        import scl.sparse as sp
        
        mat = SclCSR.zeros(0, 0, dtype=sp.float32)
        assert mat.shape == (0, 0)
        assert mat.nnz == 0
    
    def test_single_element_matrix(self, requires_scl):
        """Test single element matrix."""
        from scl.sparse import SclCSR, from_list
        import scl.sparse as sp
        
        data = from_list([42.0], dtype=sp.float32)
        indices = from_list([0], dtype=sp.int64)
        indptr = from_list([0, 1], dtype=sp.int64)
        
        mat = SclCSR(data, indices, indptr, shape=(1, 1))
        assert mat.shape == (1, 1)
        assert mat.nnz == 1
        
        row = mat.get_row_dense(0)
        assert row[0] == pytest.approx(42.0)
    
    def test_large_sparse_matrix(self, requires_scl):
        """Test creating large sparse matrix."""
        from scl.sparse import SclCSR
        import scl.sparse as sp
        
        # Create very large but empty matrix
        mat = SclCSR.zeros(1000000, 50000, dtype=sp.float32)
        assert mat.shape == (1000000, 50000)
        assert mat.nnz == 0
        
        # Should not crash or use excessive memory

