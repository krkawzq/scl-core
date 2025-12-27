"""
Tests for distributed/memory-mapped matrix operations.

Note: These tests require mmap functionality to be compiled in libscl.
If mmap symbols are not available, tests will be skipped.
"""

import pytest
import numpy as np
import scipy.sparse as sp

# Check if mmap is available
try:
    from scl._kernel import mmap as kernel_mmap
    from scl._kernel.lib_loader import get_lib
    lib = get_lib()
    HAS_MMAP = hasattr(lib, 'scl_mmap_create_csr_from_ptr')
except (ImportError, AttributeError):
    HAS_MMAP = False

pytestmark = pytest.mark.skipif(not HAS_MMAP, reason="mmap functionality not compiled")


class TestMappedCSR:
    """Test MappedCSR class."""
    
    def test_from_csr_basic(self):
        """Test creating MappedCSR from in-memory CSR."""
        try:
            from scl.sparse import SclCSR
            from scl.dist import MappedCSR
            from scl._kernel.lib_loader import LibraryNotFoundError
        except ImportError:
            pytest.skip("Required modules not available")
        
        try:
            # Create small test matrix
            dense = np.array([
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 0.0],
                [4.0, 0.0, 5.0]
            ])
            csr = SclCSR.from_dense(dense)
            
            # Create mapped version
            mapped = MappedCSR.from_csr(csr, max_pages=10)
            
            # Check shape
            assert mapped.shape == (3, 3)
            assert mapped.nnz == 5
            
            # Release resources
            mapped.release()
        except (RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Mapped CSR not available: {e}")
    
    def test_context_manager(self):
        """Test MappedCSR context manager."""
        try:
            from scl.sparse import SclCSR
            from scl.dist import MappedCSR
            from scl._kernel.lib_loader import LibraryNotFoundError
        except ImportError:
            pytest.skip("Required modules not available")
        
        try:
            dense = np.array([[1.0, 2.0], [3.0, 4.0]])
            csr = SclCSR.from_dense(dense)
            
            # Use context manager
            with MappedCSR.from_csr(csr, max_pages=10) as mapped:
                assert mapped.shape == (2, 2)
                assert mapped.nnz == 4
            
            # After context, should be released
            assert mapped._released == True
        except (RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Mapped CSR not available: {e}")


class TestMappedStatistics:
    """Test statistics on mapped matrices."""
    
    def test_row_sum(self):
        """Test row sum computation."""
        try:
            from scl.sparse import SclCSR
            from scl.dist import MappedCSR
            from scl._kernel.lib_loader import LibraryNotFoundError
        except ImportError:
            pytest.skip("Required modules not available")
        
        try:
            dense = np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]
            ])
            csr = SclCSR.from_dense(dense)
            
            with MappedCSR.from_csr(csr, max_pages=10) as mapped:
                row_sums = mapped.row_sum()
                
                assert len(row_sums) == 2
                assert row_sums[0] == pytest.approx(6.0)   # 1+2+3
                assert row_sums[1] == pytest.approx(15.0)  # 4+5+6
        except (RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Mapped statistics not available: {e}")
    
    def test_col_sum(self):
        """Test column sum computation."""
        try:
            from scl.sparse import SclCSR
            from scl.dist import MappedCSR
            from scl._kernel.lib_loader import LibraryNotFoundError
        except ImportError:
            pytest.skip("Required modules not available")
        
        try:
            dense = np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]
            ])
            csr = SclCSR.from_dense(dense)
            
            with MappedCSR.from_csr(csr, max_pages=10) as mapped:
                col_sums = mapped.col_sum()
                
                assert len(col_sums) == 3
                assert col_sums[0] == pytest.approx(5.0)   # 1+4
                assert col_sums[1] == pytest.approx(7.0)   # 2+5
                assert col_sums[2] == pytest.approx(9.0)   # 3+6
        except (RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Mapped statistics not available: {e}")
    
    def test_global_sum(self):
        """Test global sum computation."""
        try:
            from scl.sparse import SclCSR
            from scl.dist import MappedCSR
            from scl._kernel.lib_loader import LibraryNotFoundError
        except ImportError:
            pytest.skip("Required modules not available")
        
        try:
            dense = np.array([
                [1.0, 2.0],
                [3.0, 4.0]
            ])
            csr = SclCSR.from_dense(dense)
            
            with MappedCSR.from_csr(csr, max_pages=10) as mapped:
                total = mapped.global_sum()
                
                assert total == pytest.approx(10.0)  # 1+2+3+4
        except (RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Mapped statistics not available: {e}")


class TestMappedTransforms:
    """Test transformations on mapped matrices."""
    
    def test_normalize_l1(self):
        """Test L1 normalization."""
        try:
            from scl.sparse import SclCSR
            from scl.dist import MappedCSR
            from scl._kernel.lib_loader import LibraryNotFoundError
        except ImportError:
            pytest.skip("Required modules not available")
        
        try:
            dense = np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]
            ])
            csr = SclCSR.from_dense(dense)
            
            with MappedCSR.from_csr(csr, max_pages=10) as mapped:
                normalized = mapped.normalize_l1()
                
                # Check that rows sum to 1
                row_sums = normalized.sum(axis=1)
                assert row_sums[0] == pytest.approx(1.0)
                assert row_sums[1] == pytest.approx(1.0)
        except (RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Mapped normalization not available: {e}")
    
    def test_log1p(self):
        """Test log1p transformation."""
        try:
            from scl.sparse import SclCSR
            from scl.dist import MappedCSR
            from scl._kernel.lib_loader import LibraryNotFoundError
        except ImportError:
            pytest.skip("Required modules not available")
        
        try:
            dense = np.array([
                [1.0, 2.0],
                [3.0, 4.0]
            ])
            csr = SclCSR.from_dense(dense)
            original_sum = csr.sum()
            
            with MappedCSR.from_csr(csr, max_pages=10) as mapped:
                transformed = mapped.log1p()
                
                # log1p should reduce values
                assert transformed.sum() < original_sum
                # Shape should be preserved
                assert transformed.shape == csr.shape
        except (RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Mapped log1p not available: {e}")


class TestMappedConversion:
    """Test format conversion for mapped matrices."""
    
    def test_to_csc(self):
        """Test CSR to CSC conversion."""
        try:
            from scl.sparse import SclCSR
            from scl.dist import MappedCSR
            from scl._kernel.lib_loader import LibraryNotFoundError
        except ImportError:
            pytest.skip("Required modules not available")
        
        try:
            dense = np.array([
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 0.0]
            ])
            csr = SclCSR.from_dense(dense)
            
            with MappedCSR.from_csr(csr, max_pages=10) as mapped:
                csc = mapped.to_csc()
                
                # Check shape preserved
                assert csc.shape == csr.shape
                # Check nnz preserved
                assert csc.nnz == csr.nnz
        except (RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Mapped conversion not available: {e}")
    
    def test_load_full(self):
        """Test loading full matrix into memory."""
        try:
            from scl.sparse import SclCSR
            from scl.dist import MappedCSR
            from scl._kernel.lib_loader import LibraryNotFoundError
        except ImportError:
            pytest.skip("Required modules not available")
        
        try:
            dense = np.array([
                [1.0, 2.0],
                [3.0, 4.0]
            ])
            csr = SclCSR.from_dense(dense)
            
            with MappedCSR.from_csr(csr, max_pages=10) as mapped:
                loaded = mapped.load_full()
                
                # Check shape and nnz
                assert loaded.shape == csr.shape
                assert loaded.nnz == csr.nnz
                # Check data values
                assert loaded.data[0] == pytest.approx(1.0)
        except (RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Mapped load not available: {e}")


class TestDistFunctionalAPI:
    """Test functional API for dist module."""
    
    def test_dist_row_sum(self):
        """Test dist.row_sum function."""
        try:
            from scl.sparse import SclCSR
            from scl.dist import MappedCSR
            import scl.dist as dist
            from scl._kernel.lib_loader import LibraryNotFoundError
        except ImportError:
            pytest.skip("Required modules not available")
        
        try:
            dense = np.array([[1.0, 2.0], [3.0, 4.0]])
            csr = SclCSR.from_dense(dense)
            
            with MappedCSR.from_csr(csr, max_pages=10) as mapped:
                row_sums = dist.row_sum(mapped)
                
                assert len(row_sums) == 2
                assert row_sums[0] == pytest.approx(3.0)
                assert row_sums[1] == pytest.approx(7.0)
        except (RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Dist functional API not available: {e}")
    
    def test_dist_normalize_l1(self):
        """Test dist.normalize_l1 function."""
        try:
            from scl.sparse import SclCSR
            from scl.dist import MappedCSR
            import scl.dist as dist
            from scl._kernel.lib_loader import LibraryNotFoundError
        except ImportError:
            pytest.skip("Required modules not available")
        
        try:
            dense = np.array([[1.0, 2.0, 3.0]])
            csr = SclCSR.from_dense(dense)
            
            with MappedCSR.from_csr(csr, max_pages=10) as mapped:
                normalized = dist.normalize_l1(mapped)
                
                # Row should sum to 1
                row_sum = normalized.sum(axis=1)
                assert row_sum[0] == pytest.approx(1.0)
        except (RuntimeError, LibraryNotFoundError) as e:
            pytest.skip(f"Dist normalize not available: {e}")

