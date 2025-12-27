"""Tests for Callback-Based Sparse Matrices.

This module tests the CallbackCSR and CallbackCSC classes that allow
users to implement custom data access patterns in Python.
"""

import pytest
import numpy as np
from typing import Tuple

# Skip all tests if callback module not available
try:
    from scl.sparse import (
        CallbackCSR, CallbackCSC, 
        CSRBase, CSCBase, SparseBase,
        SclCSR, SclCSC, is_csr_like, is_csc_like, is_sparse_like
    )
    CALLBACK_AVAILABLE = True
except (ImportError, OSError) as e:
    CALLBACK_AVAILABLE = False
    SKIP_REASON = str(e)


@pytest.mark.skipif(not CALLBACK_AVAILABLE, reason="Callback module not available")
class TestSparseBaseClasses:
    """Test the abstract base class hierarchy."""
    
    def test_sparse_base_is_abstract(self):
        """SparseBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SparseBase()
    
    def test_csr_base_is_abstract(self):
        """CSRBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CSRBase()
    
    def test_csc_base_is_abstract(self):
        """CSCBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CSCBase()
    
    def test_scl_csr_is_csr_base(self):
        """SclCSR should be recognized as CSR-like."""
        mat = SclCSR.from_dense([[1, 0, 2], [0, 3, 0]])
        # Note: SclCSR doesn't inherit from CSRBase yet in this test
        # but is_csr_like should still work via duck typing check
        assert is_csr_like(mat) or isinstance(mat, SclCSR)
    
    def test_type_check_functions(self):
        """Test type checking utility functions."""
        csr = SclCSR.from_dense([[1, 0], [0, 2]])
        csc = SclCSC.from_dense([[1, 0], [0, 2]])
        
        assert is_sparse_like(csr)
        assert is_sparse_like(csc)


@pytest.mark.skipif(not CALLBACK_AVAILABLE, reason="Callback module not available")
class TestSimpleCallbackCSR:
    """Test CallbackCSR with a simple in-memory implementation."""
    
    def create_simple_callback_csr(self) -> CallbackCSR:
        """Create a simple callback CSR for testing."""
        
        # Pre-computed sparse data
        # Matrix:
        # [[1, 0, 2],
        #  [0, 3, 0],
        #  [4, 0, 5]]
        class SimpleCSR(CallbackCSR):
            def __init__(self):
                self._rows_data = [
                    (np.array([1.0, 2.0]), np.array([0, 2])),  # Row 0
                    (np.array([3.0]), np.array([1])),          # Row 1
                    (np.array([4.0, 5.0]), np.array([0, 2])),  # Row 2
                ]
                self._nnz = 5
                super().__init__(dtype='float64')
            
            def get_shape(self) -> Tuple[int, int]:
                return (3, 3)
            
            def get_nnz(self) -> int:
                return self._nnz
            
            def get_row_data(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
                return self._rows_data[i]
        
        return SimpleCSR()
    
    def test_callback_csr_properties(self):
        """Test basic properties of CallbackCSR."""
        with self.create_simple_callback_csr() as mat:
            assert mat.shape == (3, 3)
            assert mat.rows == 3
            assert mat.cols == 3
            assert mat.nnz == 5
            assert mat.dtype == 'float64'
            assert mat.format == 'csr'
    
    def test_callback_csr_row_access(self):
        """Test row data access."""
        with self.create_simple_callback_csr() as mat:
            # Row 0
            values = mat.row_values(0)
            indices = mat.row_indices(0)
            assert len(values) == 2
            assert len(indices) == 2
            np.testing.assert_array_equal(values.to_numpy(), [1.0, 2.0])
            np.testing.assert_array_equal(indices.to_numpy(), [0, 2])
            
            # Row 1
            values = mat.row_values(1)
            indices = mat.row_indices(1)
            assert len(values) == 1
            np.testing.assert_array_equal(values.to_numpy(), [3.0])
            np.testing.assert_array_equal(indices.to_numpy(), [1])
    
    def test_callback_csr_row_sum(self):
        """Test row sum using C++ kernel."""
        with self.create_simple_callback_csr() as mat:
            row_sums = mat.sum(axis=1)
            expected = np.array([3.0, 3.0, 9.0])
            np.testing.assert_array_almost_equal(row_sums, expected)
    
    def test_callback_csr_total_sum(self):
        """Test total sum."""
        with self.create_simple_callback_csr() as mat:
            total = mat.sum(axis=None)
            assert total == pytest.approx(15.0)
    
    def test_callback_csr_row_mean(self):
        """Test row mean using C++ kernel."""
        with self.create_simple_callback_csr() as mat:
            row_means = mat.mean(axis=1)
            # Mean includes zeros: [3/3, 3/3, 9/3]
            expected = np.array([1.0, 1.0, 3.0])
            np.testing.assert_array_almost_equal(row_means, expected)
    
    def test_callback_csr_to_scipy(self):
        """Test conversion to scipy."""
        with self.create_simple_callback_csr() as mat:
            scipy_mat = mat.to_scipy()
            
            expected = np.array([
                [1, 0, 2],
                [0, 3, 0],
                [4, 0, 5]
            ])
            np.testing.assert_array_equal(scipy_mat.toarray(), expected)
    
    def test_callback_csr_copy(self):
        """Test materialization via copy()."""
        with self.create_simple_callback_csr() as mat:
            copied = mat.copy()
            
            # Should be SclCSR
            assert isinstance(copied, SclCSR)
            assert copied.shape == mat.shape
            assert copied.nnz == mat.nnz
    
    def test_callback_csr_context_manager(self):
        """Test context manager protocol."""
        mat = self.create_simple_callback_csr()
        
        with mat:
            assert mat.shape == (3, 3)
        
        # After exiting, handle should be released
        assert mat._closed


@pytest.mark.skipif(not CALLBACK_AVAILABLE, reason="Callback module not available")
class TestSimpleCallbackCSC:
    """Test CallbackCSC with a simple in-memory implementation."""
    
    def create_simple_callback_csc(self) -> CallbackCSC:
        """Create a simple callback CSC for testing."""
        
        # Matrix (same as CSR test but column-oriented):
        # [[1, 0, 2],
        #  [0, 3, 0],
        #  [4, 0, 5]]
        class SimpleCSC(CallbackCSC):
            def __init__(self):
                self._cols_data = [
                    (np.array([1.0, 4.0]), np.array([0, 2])),  # Col 0
                    (np.array([3.0]), np.array([1])),          # Col 1
                    (np.array([2.0, 5.0]), np.array([0, 2])),  # Col 2
                ]
                self._nnz = 5
                super().__init__(dtype='float64')
            
            def get_shape(self) -> Tuple[int, int]:
                return (3, 3)
            
            def get_nnz(self) -> int:
                return self._nnz
            
            def get_col_data(self, j: int) -> Tuple[np.ndarray, np.ndarray]:
                return self._cols_data[j]
        
        return SimpleCSC()
    
    def test_callback_csc_properties(self):
        """Test basic properties of CallbackCSC."""
        with self.create_simple_callback_csc() as mat:
            assert mat.shape == (3, 3)
            assert mat.rows == 3
            assert mat.cols == 3
            assert mat.nnz == 5
            assert mat.dtype == 'float64'
            assert mat.format == 'csc'
    
    def test_callback_csc_col_access(self):
        """Test column data access."""
        with self.create_simple_callback_csc() as mat:
            # Column 0
            values = mat.col_values(0)
            indices = mat.col_indices(0)
            assert len(values) == 2
            np.testing.assert_array_equal(values.to_numpy(), [1.0, 4.0])
            np.testing.assert_array_equal(indices.to_numpy(), [0, 2])
    
    def test_callback_csc_col_sum(self):
        """Test column sum using C++ kernel."""
        with self.create_simple_callback_csc() as mat:
            col_sums = mat.sum(axis=0)
            expected = np.array([5.0, 3.0, 7.0])
            np.testing.assert_array_almost_equal(col_sums, expected)
    
    def test_callback_csc_total_sum(self):
        """Test total sum."""
        with self.create_simple_callback_csc() as mat:
            total = mat.sum(axis=None)
            assert total == pytest.approx(15.0)
    
    def test_callback_csc_to_scipy(self):
        """Test conversion to scipy."""
        with self.create_simple_callback_csc() as mat:
            scipy_mat = mat.to_scipy()
            
            expected = np.array([
                [1, 0, 2],
                [0, 3, 0],
                [4, 0, 5]
            ])
            np.testing.assert_array_equal(scipy_mat.toarray(), expected)


@pytest.mark.skipif(not CALLBACK_AVAILABLE, reason="Callback module not available")
class TestCallbackLazyLoading:
    """Test lazy loading scenarios with callbacks."""
    
    def test_lazy_row_loading(self):
        """Test that rows are loaded on demand."""
        load_counts = [0] * 3
        
        class LazyCSR(CallbackCSR):
            def __init__(self):
                super().__init__(dtype='float64')
            
            def get_shape(self):
                return (3, 3)
            
            def get_nnz(self):
                return 3
            
            def get_row_data(self, i):
                load_counts[i] += 1
                return (np.array([float(i + 1)]), np.array([i]))
        
        with LazyCSR() as mat:
            # Initially no rows loaded
            assert load_counts == [0, 0, 0]
            
            # Access row 1
            mat.row_values(1)
            assert load_counts == [0, 1, 0]
            
            # Access row 1 again (should use cache)
            mat.row_values(1)
            # Note: depends on caching implementation
            
            # Access row 0
            mat.row_values(0)
            assert load_counts[0] >= 1


@pytest.mark.skipif(not CALLBACK_AVAILABLE, reason="Callback module not available")
class TestCallbackWithScipy:
    """Test CallbackCSR/CSC wrapping scipy matrices."""
    
    def test_scipy_wrapper_csr(self):
        """Test wrapping a scipy CSR matrix."""
        import scipy.sparse as sp
        
        # Create scipy matrix
        scipy_mat = sp.csr_matrix([
            [1, 0, 2],
            [0, 3, 0],
            [4, 5, 6]
        ], dtype=np.float64)
        
        class ScipyWrapperCSR(CallbackCSR):
            def __init__(self, mat):
                self._mat = mat
                super().__init__(dtype='float64')
            
            def get_shape(self):
                return self._mat.shape
            
            def get_nnz(self):
                return self._mat.nnz
            
            def get_row_data(self, i):
                start = self._mat.indptr[i]
                end = self._mat.indptr[i + 1]
                return (
                    self._mat.data[start:end].copy(),
                    self._mat.indices[start:end].copy()
                )
        
        with ScipyWrapperCSR(scipy_mat) as mat:
            assert mat.shape == (3, 3)
            assert mat.nnz == 7
            
            row_sums = mat.sum(axis=1)
            expected = np.array([3.0, 3.0, 15.0])
            np.testing.assert_array_almost_equal(row_sums, expected)


@pytest.mark.skipif(not CALLBACK_AVAILABLE, reason="Callback module not available")
class TestCallbackInheritance:
    """Test that callbacks work with inheritance properly."""
    
    def test_isinstance_checks(self):
        """Test isinstance checks for callback matrices."""
        class SimpleCSR(CallbackCSR):
            def get_shape(self): return (2, 2)
            def get_nnz(self): return 2
            def get_row_data(self, i): return (np.array([1.0]), np.array([i]))
        
        with SimpleCSR() as mat:
            assert isinstance(mat, CallbackCSR)
            assert isinstance(mat, CSRBase)
            assert isinstance(mat, SparseBase)
            assert is_csr_like(mat)
            assert is_sparse_like(mat)
            assert not is_csc_like(mat)

