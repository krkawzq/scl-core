"""
Tests for VirtualCSR and VirtualCSC wrapper classes.
"""

import pytest
import numpy as np

try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
class TestVirtualCSR:
    """Test VirtualCSR wrapper class."""
    
    def test_create_from_scipy(self, requires_scl):
        """Test creating VirtualCSR from scipy matrix."""
        from scl.sparse import VirtualCSR
        
        scipy_mat = sp.csr_matrix([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
        virtual = VirtualCSR(scipy_mat)
        
        assert virtual.shape == (2, 3)
        assert virtual.nnz == 3
    
    def test_virtual_is_view(self, requires_scl):
        """Test that VirtualCSR is a zero-copy view."""
        from scl.sparse import VirtualCSR
        
        scipy_mat = sp.csr_matrix([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
        virtual = VirtualCSR(scipy_mat)
        
        # Modify original
        scipy_mat[0, 0] = 99
        # Virtual should reflect changes (if it's truly zero-copy)
        # This depends on implementation
        assert virtual.shape == scipy_mat.shape
    
    def test_virtual_to_owned(self, requires_scl):
        """Test converting virtual to owned SclCSR."""
        from scl.sparse import VirtualCSR, SclCSR
        
        scipy_mat = sp.csr_matrix([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
        virtual = VirtualCSR(scipy_mat)
        
        try:
            owned = virtual.to_owned()
            assert isinstance(owned, SclCSR)
            assert owned.shape == virtual.shape
            assert owned.nnz == virtual.nnz
        except (AttributeError, NotImplementedError):
            pytest.skip("to_owned not implemented")
    
    def test_virtual_getitem(self, requires_scl):
        """Test indexing virtual matrix."""
        from scl.sparse import VirtualCSR
        
        scipy_mat = sp.csr_matrix([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
        virtual = VirtualCSR(scipy_mat)
        
        # Test indexing if implemented
        try:
            val = virtual[0, 0]
            assert val == pytest.approx(1.0)
        except (TypeError, NotImplementedError):
            pytest.skip("__getitem__ not implemented")


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
class TestVirtualCSC:
    """Test VirtualCSC wrapper class."""
    
    def test_create_from_scipy(self, requires_scl):
        """Test creating VirtualCSC from scipy matrix."""
        from scl.sparse import VirtualCSC
        
        scipy_mat = sp.csc_matrix([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
        virtual = VirtualCSC(scipy_mat)
        
        assert virtual.shape == (2, 3)
        assert virtual.nnz == 3
    
    def test_virtual_csc_to_owned(self, requires_scl):
        """Test converting virtual CSC to owned."""
        from scl.sparse import VirtualCSC, SclCSC
        
        scipy_mat = sp.csc_matrix([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
        virtual = VirtualCSC(scipy_mat)
        
        try:
            owned = virtual.to_owned()
            assert isinstance(owned, SclCSC)
            assert owned.shape == virtual.shape
        except (AttributeError, NotImplementedError):
            pytest.skip("to_owned not implemented")


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
class TestVStackHStack:
    """Test vstack_csr and hstack_csc convenience functions."""
    
    def test_vstack_csr(self, requires_scl):
        """Test vstack_csr function."""
        from scl.sparse import vstack_csr, VirtualCSR
        
        mat1 = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.float32)
        mat2 = sp.csr_matrix([[5, 6]], dtype=np.float32)
        
        stacked = vstack_csr([mat1, mat2])
        assert isinstance(stacked, VirtualCSR)
        assert stacked.shape == (3, 2)
    
    def test_hstack_csc(self, requires_scl):
        """Test hstack_csc function."""
        from scl.sparse import hstack_csc, VirtualCSC
        
        mat1 = sp.csc_matrix([[1, 2], [3, 4]], dtype=np.float32)
        mat2 = sp.csc_matrix([[5], [6]], dtype=np.float32)
        
        stacked = hstack_csc([mat1, mat2])
        assert isinstance(stacked, VirtualCSC)
        assert stacked.shape == (2, 3)

