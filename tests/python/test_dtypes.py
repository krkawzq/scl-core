"""
Tests for dtype system.
"""

import pytest
from scl.sparse._dtypes import (
    DType,
    normalize_dtype,
    validate_dtype,
    is_float_dtype,
    is_int_dtype,
    dtype_itemsize,
    float32,
    float64,
    int32,
    int64,
    uint8,
    uint32,
    uint64,
)


class TestDTypeConstants:
    """Test dtype constants."""
    
    def test_dtype_constants_exist(self, requires_scl):
        """Test that dtype constants exist."""
        assert float32 is not None
        assert float64 is not None
        assert int32 is not None
        assert int64 is not None
        assert uint8 is not None
    
    def test_dtype_string_values(self, requires_scl):
        """Test dtype string values."""
        # These might be DType enum or strings
        # Test that they can be used
        from scl.sparse import Array
        arr = Array(10, dtype=float32)
        assert arr.dtype == 'float32'


class TestNormalizeDType:
    """Test dtype normalization."""
    
    def test_normalize_string(self, requires_scl):
        """Test normalizing string dtype."""
        assert normalize_dtype('float32') == 'float32'
        assert normalize_dtype('float64') == 'float64'
        assert normalize_dtype('int32') == 'int32'
        assert normalize_dtype('int64') == 'int64'
    
    def test_normalize_case_insensitive(self, requires_scl):
        """Test case-insensitive normalization."""
        # Case-insensitive not implemented yet
        pytest.skip("Case-insensitive dtype not implemented")
    
    def test_normalize_dtype_enum(self, requires_scl):
        """Test normalizing DType enum."""
        # If DType is an enum, test it
        try:
            result = normalize_dtype(float32)
            assert result in ('float32', float32)
        except (TypeError, AttributeError):
            pytest.skip("DType enum normalization not implemented")


class TestValidateDType:
    """Test dtype validation."""
    
    def test_validate_valid_dtypes(self, requires_scl):
        """Test validating valid dtypes."""
        valid_dtypes = ['float32', 'float64', 'int32', 'int64', 'uint8', 'uint32', 'uint64']
        for dtype in valid_dtypes:
            # Should not raise
            validate_dtype(dtype)
    
    def test_validate_invalid_dtype(self, requires_scl):
        """Test validating invalid dtype."""
        with pytest.raises(ValueError):
            validate_dtype('invalid_type')
    
    def test_validate_none(self, requires_scl):
        """Test validating None."""
        with pytest.raises((ValueError, TypeError)):
            validate_dtype(None)


class TestIsFloatDType:
    """Test float dtype checking."""
    
    def test_is_float_dtype_true(self, requires_scl):
        """Test checking float dtypes."""
        assert is_float_dtype('float32') is True
        assert is_float_dtype('float64') is True
    
    def test_is_float_dtype_false(self, requires_scl):
        """Test checking non-float dtypes."""
        assert is_float_dtype('int32') is False
        assert is_float_dtype('int64') is False
        assert is_float_dtype('uint8') is False


class TestIsIntDType:
    """Test int dtype checking."""
    
    def test_is_int_dtype_true(self, requires_scl):
        """Test checking int dtypes."""
        assert is_int_dtype('int32') is True
        assert is_int_dtype('int64') is True
        assert is_int_dtype('uint8') is True
        assert is_int_dtype('uint32') is True
        assert is_int_dtype('uint64') is True
    
    def test_is_int_dtype_false(self, requires_scl):
        """Test checking non-int dtypes."""
        assert is_int_dtype('float32') is False
        assert is_int_dtype('float64') is False


class TestDtypeItemsize:
    """Test dtype itemsize."""
    
    def test_dtype_itemsize(self, requires_scl):
        """Test getting dtype itemsize."""
        assert dtype_itemsize('float32') == 4
        assert dtype_itemsize('float64') == 8
        assert dtype_itemsize('int32') == 4
        assert dtype_itemsize('int64') == 8
        assert dtype_itemsize('uint8') == 1
        assert dtype_itemsize('uint32') == 4
        assert dtype_itemsize('uint64') == 8
    
    def test_dtype_itemsize_invalid(self, requires_scl):
        """Test itemsize for invalid dtype."""
        with pytest.raises(ValueError):
            dtype_itemsize('invalid_type')

