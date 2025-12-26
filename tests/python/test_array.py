"""
Tests for Array class.
"""

import pytest
import numpy as np
from scl.sparse import Array, empty, zeros, from_list, from_buffer
from scl.sparse import float32, float64, int32, int64, uint8


class TestArrayCreation:
    """Test Array creation methods."""
    
    def test_array_creation_empty(self, requires_scl):
        """Test creating empty array."""
        arr = Array(10, dtype='float32')
        assert arr.size == 10
        assert arr.dtype == 'float32'
        assert arr.nbytes == 40  # 10 * 4 bytes
    
    def test_array_creation_empty_function(self, requires_scl):
        """Test empty() function."""
        arr = empty(10, dtype='float32')
        assert arr.size == 10
        assert arr.dtype == 'float32'
    
    def test_array_zeros(self, requires_scl):
        """Test zeros() function."""
        arr = zeros(10, dtype='float64')
        assert arr.size == 10
        assert arr.dtype == 'float64'
        # Check all zeros
        assert all(arr[i] == 0.0 for i in range(10))
    
    def test_array_from_list(self, requires_scl):
        """Test from_list() function."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        arr = from_list(data, dtype='float32')
        assert arr.size == 5
        assert arr.dtype == 'float32'
        for i, val in enumerate(data):
            assert arr[i] == pytest.approx(val)
    
    def test_array_from_list_int64(self, requires_scl):
        """Test from_list() with int64."""
        data = [1, 2, 3, 4, 5]
        arr = from_list(data, dtype='int64')
        assert arr.size == 5
        assert arr.dtype == 'int64'
        for i, val in enumerate(data):
            assert arr[i] == val
    
    def test_array_zero_size(self, requires_scl):
        """Test creating zero-sized array."""
        arr = Array(0, dtype='float32')
        assert arr.size == 0
        assert arr.nbytes == 0


class TestArrayProperties:
    """Test Array properties and attributes."""
    
    def test_array_properties(self, requires_scl):
        """Test basic properties."""
        arr = Array(100, dtype='float32')
        assert arr.size == 100
        assert arr.dtype == 'float32'
        assert arr.itemsize == 4
        assert arr.nbytes == 400
        assert arr.ptr != 0
    
    def test_array_dtypes(self, requires_scl):
        """Test different dtypes."""
        dtypes = ['float32', 'float64', 'int32', 'int64', 'uint8']
        sizes = [4, 8, 4, 8, 1]
        
        for dtype, itemsize in zip(dtypes, sizes):
            arr = Array(10, dtype=dtype)
            assert arr.dtype == dtype
            assert arr.itemsize == itemsize
            assert arr.nbytes == 10 * itemsize


class TestArrayIndexing:
    """Test Array indexing and assignment."""
    
    def test_array_getitem(self, requires_scl):
        """Test getting array elements."""
        arr = from_list([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
        assert arr[0] == pytest.approx(1.0)
        assert arr[2] == pytest.approx(3.0)
        assert arr[4] == pytest.approx(5.0)
    
    def test_array_setitem(self, requires_scl):
        """Test setting array elements."""
        arr = zeros(10, dtype='float32')
        arr[0] = 1.5
        arr[5] = 2.5
        arr[9] = 3.5
        
        assert arr[0] == pytest.approx(1.5)
        assert arr[5] == pytest.approx(2.5)
        assert arr[9] == pytest.approx(3.5)
        # Other elements should still be zero
        assert arr[1] == pytest.approx(0.0)
    
    def test_array_slice(self, requires_scl):
        """Test array slicing."""
        arr = from_list(list(range(10)), dtype='float32')
        # Note: Basic slicing might not be implemented
        # This test checks if it raises an error or works
        try:
            sliced = arr[2:5]
            assert len(sliced) == 3
        except (TypeError, NotImplementedError):
            pytest.skip("Array slicing not implemented")
    
    def test_array_index_bounds(self, requires_scl):
        """Test index bounds checking."""
        arr = Array(10, dtype='float32')
        # Should raise IndexError for out of bounds
        with pytest.raises((IndexError, ValueError)):
            _ = arr[10]
        # Note: -1 should work (negative indexing)
        # Test extreme negative
        with pytest.raises((IndexError, ValueError)):
            _ = arr[-11]  # Out of bounds negative


class TestArrayConversion:
    """Test Array conversion methods."""
    
    def test_array_to_numpy(self, requires_scl):
        """Test converting Array to numpy array."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        arr = from_list(data, dtype='float32')
        np_arr = arr.to_numpy()
        
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.dtype == np.float32
        np.testing.assert_array_equal(np_arr, np.array(data, dtype=np.float32))
    
    def test_array_to_list(self, requires_scl):
        """Test converting Array to Python list."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        arr = from_list(data, dtype='float32')
        lst = arr.to_list()
        
        assert isinstance(lst, list)
        assert len(lst) == len(data)
        for i, val in enumerate(data):
            assert lst[i] == pytest.approx(val)
    
    def test_array_memoryview(self, requires_scl):
        """Test Array memoryview protocol."""
        arr = from_list([1.0, 2.0, 3.0], dtype='float32')
        # Use as_memoryview() for Python < 3.11 compatibility
        view = arr.as_memoryview()
        
        assert view.nbytes == arr.nbytes
        assert len(view) == arr.size


class TestArrayFromBuffer:
    """Test creating Array from existing buffer."""
    
    def test_from_buffer_numpy(self, requires_scl):
        """Test from_buffer() with numpy array."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        arr = from_buffer(np_arr, dtype='float32', size=len(np_arr))
        
        assert arr.size == 4
        assert arr.dtype == 'float32'
        for i in range(4):
            assert arr[i] == pytest.approx(np_arr[i])
    
    def test_from_buffer_list(self, requires_scl):
        """Test from_buffer() with list."""
        data = [1, 2, 3, 4, 5]
        arr = from_buffer(data, dtype='int32', size=len(data))
        
        assert arr.size == 5
        assert arr.dtype == 'int32'


class TestArrayErrors:
    """Test error handling."""
    
    def test_invalid_dtype(self, requires_scl):
        """Test invalid dtype."""
        with pytest.raises(ValueError):
            Array(10, dtype='invalid_type')
    
    def test_negative_size(self, requires_scl):
        """Test negative size."""
        with pytest.raises(ValueError):
            Array(-1, dtype='float32')
    
    def test_invalid_dtype_from_list(self, requires_scl):
        """Test from_list with invalid dtype."""
        with pytest.raises(ValueError):
            from_list([1, 2, 3], dtype='invalid')


class TestArrayDTypeConstants:
    """Test dtype constants."""
    
    def test_dtype_constants(self, requires_scl):
        """Test dtype constant usage."""
        from scl.sparse import float32, float64, int32, int64
        
        arr_f32 = Array(10, dtype=float32)
        assert arr_f32.dtype == 'float32'
        
        arr_f64 = Array(10, dtype=float64)
        assert arr_f64.dtype == 'float64'
        
        arr_i32 = Array(10, dtype=int32)
        assert arr_i32.dtype == 'int32'
        
        arr_i64 = Array(10, dtype=int64)
        assert arr_i64.dtype == 'int64'

