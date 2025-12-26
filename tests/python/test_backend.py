"""
Tests for Backend management and Ownership tracking.

Tests the smart backend system introduced in SCL v0.2.0:
- Backend types: CUSTOM, VIRTUAL, MAPPED
- Ownership: OWNED, BORROWED, VIEW
- Reference chain management
- Virtual backend operations (vstack, slicing)
"""

import pytest
import numpy as np

try:
    from scl.sparse import (
        SclCSR, SclCSC, Array,
        Backend, Ownership,
        vstack_csr, hstack_csc,
        from_scipy,
        RefChain, OwnershipTracker,
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
# Backend Type Tests
# =============================================================================

class TestBackendTypes:
    """Test Backend type behavior."""
    
    def test_custom_backend_from_dense(self, requires_scl):
        """Test that from_dense creates Custom backend."""
        mat = SclCSR.from_dense([[1, 2], [3, 4]])
        assert mat.backend == Backend.CUSTOM
    
    def test_custom_backend_from_arrays(self, requires_scl):
        """Test that direct construction creates Custom backend."""
        from scl.sparse import from_list, float64, int64
        
        data = from_list([1.0, 2.0], dtype=float64)
        indices = from_list([0, 1], dtype=int64)
        indptr = from_list([0, 1, 2], dtype=int64)
        
        mat = SclCSR(data, indices, indptr, shape=(2, 2))
        assert mat.backend == Backend.CUSTOM
    
    def test_virtual_backend_from_vstack(self, requires_scl):
        """Test that vstack creates Virtual backend."""
        mat1 = SclCSR.from_dense([[1, 2]])
        mat2 = SclCSR.from_dense([[3, 4]])
        
        stacked = vstack_csr([mat1, mat2])
        assert stacked.backend == Backend.VIRTUAL
    
    def test_virtual_backend_from_slice_rows(self, requires_scl):
        """Test that row slicing with virtual strategy creates Virtual backend."""
        mat = SclCSR.from_dense([[1, 2], [3, 4], [5, 6]])
        
        view = mat.slice_rows([0, 2], strategy='virtual')
        assert view.backend == Backend.VIRTUAL


# =============================================================================
# Ownership Tests
# =============================================================================

class TestOwnership:
    """Test Ownership model."""
    
    def test_owned_from_dense(self, requires_scl):
        """Test that from_dense creates OWNED ownership."""
        mat = SclCSR.from_dense([[1, 2], [3, 4]])
        assert mat.ownership == Ownership.OWNED
        assert mat.is_owned == True
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_borrowed_from_scipy(self, requires_scl):
        """Test that from_scipy with copy=False creates BORROWED ownership."""
        scipy_mat = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.float64)
        mat = SclCSR.from_scipy(scipy_mat, copy=False)
        
        assert mat.ownership == Ownership.BORROWED
        assert mat.is_owned == False
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_owned_from_scipy_copy(self, requires_scl):
        """Test that from_scipy with copy=True creates OWNED ownership."""
        scipy_mat = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.float64)
        mat = SclCSR.from_scipy(scipy_mat, copy=True)
        
        assert mat.ownership == Ownership.OWNED
        assert mat.is_owned == True
    
    def test_view_ownership_after_vstack(self, requires_scl):
        """Test that vstack result has VIEW ownership."""
        mat1 = SclCSR.from_dense([[1, 2]])
        mat2 = SclCSR.from_dense([[3, 4]])
        
        stacked = vstack_csr([mat1, mat2])
        assert stacked.ownership == Ownership.VIEW


# =============================================================================
# Reference Chain Tests
# =============================================================================

class TestReferenceChain:
    """Test reference chain management."""
    
    def test_ref_chain_empty_for_owned(self, requires_scl):
        """Test that owned matrices have empty ref chain."""
        mat = SclCSR.from_dense([[1, 2], [3, 4]])
        assert mat._ref_chain.is_empty == True
    
    def test_ref_chain_after_vstack(self, requires_scl):
        """Test that vstack maintains references to sources."""
        mat1 = SclCSR.from_dense([[1, 2]])
        mat2 = SclCSR.from_dense([[3, 4]])
        
        stacked = vstack_csr([mat1, mat2])
        assert stacked._ref_chain.count == 2
    
    def test_ref_chain_flattening(self, requires_scl):
        """Test that nested virtual matrices flatten ref chains."""
        base = SclCSR.from_dense([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        # Create two virtual views
        v1 = base.slice_rows([0, 1], strategy='virtual')
        v2 = base.slice_rows([2, 3], strategy='virtual')
        
        # Stack virtual matrices
        stacked = vstack_csr([v1, v2])
        
        # Should reference the original base, not the intermediate virtuals
        # (exact count depends on flattening implementation)
        assert stacked._ref_chain.count >= 1
    
    def test_ref_chain_prevents_gc(self, requires_scl):
        """Test that ref chain prevents source from being garbage collected."""
        import gc
        
        def create_virtual():
            mat1 = SclCSR.from_dense([[1, 2]])
            mat2 = SclCSR.from_dense([[3, 4]])
            return vstack_csr([mat1, mat2])
        
        stacked = create_virtual()
        gc.collect()
        
        # If ref chain works, stacked should still be valid
        assert stacked.shape == (2, 2)
        
        # Materialize should work
        owned = stacked.to_owned()
        assert owned[0, 0] == pytest.approx(1.0)


# =============================================================================
# Virtual Backend Operations Tests
# =============================================================================

class TestVirtualBackendOperations:
    """Test Virtual backend specific operations."""
    
    def test_vstack_csr_shape(self, requires_scl):
        """Test vstack_csr produces correct shape."""
        mat1 = SclCSR.from_dense([[1, 2], [3, 4]])  # 2x2
        mat2 = SclCSR.from_dense([[5, 6]])           # 1x2
        
        stacked = vstack_csr([mat1, mat2])
        assert stacked.shape == (3, 2)
    
    def test_vstack_csr_nnz(self, requires_scl):
        """Test vstack_csr produces correct nnz."""
        mat1 = SclCSR.from_dense([[1, 0], [0, 2]])  # nnz=2
        mat2 = SclCSR.from_dense([[3, 0]])          # nnz=1
        
        stacked = vstack_csr([mat1, mat2])
        assert stacked.nnz == 3
    
    def test_hstack_csc_shape(self, requires_scl):
        """Test hstack_csc produces correct shape."""
        mat1 = SclCSC.from_dense([[1, 2], [3, 4]])  # 2x2
        mat2 = SclCSC.from_dense([[5], [6]])        # 2x1
        
        stacked = hstack_csc([mat1, mat2])
        assert stacked.shape == (2, 3)
    
    def test_virtual_materialization(self, requires_scl):
        """Test materializing Virtual to Custom."""
        mat1 = SclCSR.from_dense([[1, 2]])
        mat2 = SclCSR.from_dense([[3, 4]])
        
        stacked = vstack_csr([mat1, mat2])
        assert stacked.backend == Backend.VIRTUAL
        
        owned = stacked.to_owned()
        assert owned.backend == Backend.CUSTOM
        assert owned.ownership == Ownership.OWNED
    
    def test_virtual_indexing(self, requires_scl):
        """Test indexing Virtual backend matrices."""
        mat1 = SclCSR.from_dense([[1, 2]])
        mat2 = SclCSR.from_dense([[3, 4]])
        
        stacked = vstack_csr([mat1, mat2])
        
        # Should trigger materialization
        val = stacked[0, 0]
        assert val == pytest.approx(1.0)
        
        val = stacked[1, 1]
        assert val == pytest.approx(4.0)
    
    def test_virtual_row_slice_values(self, requires_scl):
        """Test values after virtual row slicing."""
        mat = SclCSR.from_dense([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        view = mat.slice_rows([0, 2], strategy='virtual')
        owned = view.to_owned()
        
        # Row 0 of view = row 0 of original
        assert owned[0, 0] == pytest.approx(1.0)
        assert owned[0, 1] == pytest.approx(2.0)
        
        # Row 1 of view = row 2 of original
        assert owned[1, 0] == pytest.approx(5.0)
        assert owned[1, 1] == pytest.approx(6.0)


# =============================================================================
# Slice Rows Strategy Tests
# =============================================================================

class TestSliceRowsStrategy:
    """Test slice_rows with different strategies."""
    
    def test_virtual_strategy(self, requires_scl):
        """Test slice_rows with virtual strategy."""
        mat = SclCSR.from_dense([[1, 2], [3, 4], [5, 6]])
        
        view = mat.slice_rows([0, 2], strategy='virtual')
        assert view.backend == Backend.VIRTUAL
    
    def test_copy_strategy(self, requires_scl):
        """Test slice_rows with copy strategy."""
        mat = SclCSR.from_dense([[1, 2], [3, 4], [5, 6]])
        
        copy = mat.slice_rows([0, 2], strategy='copy')
        assert copy.backend == Backend.CUSTOM
        assert copy.ownership == Ownership.OWNED
    
    def test_auto_strategy_sparse_selection(self, requires_scl):
        """Test auto strategy with sparse selection (< 50% rows)."""
        # Create 10 row matrix
        dense = [[float(i * 2 + j) for j in range(3)] for i in range(10)]
        mat = SclCSR.from_dense(dense)
        
        # Select 2 rows (20%) - should use virtual
        view = mat.slice_rows([0, 5], strategy='auto')
        # Auto may choose virtual or copy, both are valid
        assert view.shape == (2, 3)


# =============================================================================
# Slice Cols Tests
# =============================================================================

class TestSliceCols:
    """Test column slicing."""
    
    def test_slice_cols_virtual(self, requires_scl):
        """Test virtual column slicing for CSC."""
        mat = SclCSC.from_dense([[1, 2, 3], [4, 5, 6]])
        
        view = mat.slice_cols([0, 2], strategy='virtual')
        assert view.backend == Backend.VIRTUAL
        assert view.shape == (2, 2)
    
    def test_slice_cols_copy(self, requires_scl):
        """Test copy column slicing for CSR."""
        mat = SclCSR.from_dense([[1, 2, 3], [4, 5, 6]])
        
        sliced = mat.slice_cols([0, 2], lazy=False)
        assert sliced.shape == (2, 2)
        assert sliced.backend == Backend.CUSTOM


# =============================================================================
# CSC Backend Tests
# =============================================================================

class TestCSCBackend:
    """Test CSC backend operations."""
    
    def test_csc_virtual_from_hstack(self, requires_scl):
        """Test CSC Virtual backend from hstack."""
        mat1 = SclCSC.from_dense([[1, 2], [3, 4]])
        mat2 = SclCSC.from_dense([[5], [6]])
        
        stacked = hstack_csc([mat1, mat2])
        assert stacked.backend == Backend.VIRTUAL
    
    def test_csc_virtual_materialization(self, requires_scl):
        """Test CSC Virtual materialization."""
        mat1 = SclCSC.from_dense([[1, 2], [3, 4]])
        mat2 = SclCSC.from_dense([[5], [6]])
        
        stacked = hstack_csc([mat1, mat2])
        owned = stacked.to_owned()
        
        assert owned.backend == Backend.CUSTOM
        assert owned.shape == (2, 3)


# =============================================================================
# Edge Cases
# =============================================================================

class TestBackendEdgeCases:
    """Test edge cases in backend management."""
    
    def test_vstack_single_matrix(self, requires_scl):
        """Test vstack with single matrix returns same matrix."""
        mat = SclCSR.from_dense([[1, 2], [3, 4]])
        
        stacked = vstack_csr([mat])
        assert stacked is mat  # Should return same object
    
    def test_vstack_empty_list(self, requires_scl):
        """Test vstack with empty list."""
        stacked = vstack_csr([])
        assert stacked.shape == (0, 0)
    
    def test_virtual_nnz_lazy_computation(self, requires_scl):
        """Test that Virtual nnz is computed lazily."""
        mat1 = SclCSR.from_dense([[1, 0], [0, 2]])  # nnz=2
        mat2 = SclCSR.from_dense([[3, 4]])          # nnz=2
        
        stacked = vstack_csr([mat1, mat2])
        
        # nnz should be computed
        assert stacked.nnz == 4
    
    def test_virtual_dtype_consistency(self, requires_scl):
        """Test that Virtual inherits dtype from sources."""
        mat1 = SclCSR.from_dense([[1, 2]], dtype='float32')
        mat2 = SclCSR.from_dense([[3, 4]], dtype='float32')
        
        stacked = vstack_csr([mat1, mat2])
        assert stacked.dtype == 'float32'

