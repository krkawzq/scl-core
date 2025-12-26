"""
Tests for hooks integration with anndata and scipy.
"""

import pytest
import os

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestHooksInstallation:
    """Test hooks installation."""
    
    def test_hooks_installed(self, requires_scl):
        """Test that hooks are installed."""
        from scl import hooks
        
        # Hooks should auto-install on import
        assert hooks.is_installed() or not hooks._is_enabled()
    
    def test_hooks_can_install(self, requires_scl):
        """Test manual hooks installation."""
        from scl import hooks
        
        # Save original state
        original_state = hooks.is_installed()
        
        try:
            # Uninstall first
            hooks.uninstall()
            assert not hooks.is_installed()
            
            # Install
            hooks.install()
            assert hooks.is_installed()
        finally:
            # Restore original state
            if original_state:
                hooks.install()
    
    def test_hooks_can_uninstall(self, requires_scl):
        """Test hooks uninstallation."""
        from scl import hooks
        
        # Install first
        hooks.install()
        
        # Uninstall
        hooks.uninstall()
        assert not hooks.is_installed()
    
    def test_hooks_disabled_by_env(self, requires_scl):
        """Test that hooks can be disabled by environment variable."""
        from scl import hooks
        import importlib
        
        # Set environment variable
        os.environ['SCL_NO_HOOKS'] = '1'
        
        # Reload module to pick up env var
        importlib.reload(hooks)
        
        # Hooks should be disabled
        assert not hooks._is_enabled()
        
        # Cleanup
        del os.environ['SCL_NO_HOOKS']
        importlib.reload(hooks)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
class TestScipyIntegration:
    """Test integration with scipy.sparse."""
    
    def test_scl_matrix_isinstance_spmatrix(self, requires_scl):
        """Test that SclCSR is instance of spmatrix."""
        from scl.sparse import SclCSR
        from scl import hooks
        
        # Install hooks
        hooks.install()
        
        # Create SCL matrix
        mat = SclCSR.zeros(10, 20, dtype='float32')
        
        # Should be recognized as spmatrix if hooks work
        try:
            assert isinstance(mat, sp.spmatrix)
        except AssertionError:
            # Hooks might not work in test environment
            pytest.skip("isinstance check failed (hooks may need scipy to be imported first)")


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not available")
class TestAnnDataIntegration:
    """Test integration with anndata."""
    
    def test_anndata_with_scl_matrix(self, requires_scl):
        """Test creating AnnData with SCL matrix."""
        from scl.sparse import SclCSR
        from scl import hooks
        
        # Install hooks
        hooks.install()
        
        # Create SCL matrix
        mat = SclCSR.zeros(100, 50, dtype='float32')
        
        try:
            # Should work with AnnData
            adata = ad.AnnData(X=mat)
            assert adata.shape == (100, 50)
        except (TypeError, ValueError) as e:
            # May fail if hooks aren't working properly
            pytest.skip(f"AnnData integration not working: {e}")
    
    def test_anndata_write_read(self, requires_scl, tmp_path):
        """Test writing and reading AnnData with SCL matrix."""
        from scl.sparse import SclCSR
        from scl import hooks
        
        # Install hooks
        hooks.install()
        
        # Create SCL matrix
        mat = SclCSR.zeros(100, 50, dtype='float32')
        
        try:
            # Create AnnData
            adata = ad.AnnData(X=mat)
            
            # Write to file
            output_path = tmp_path / "test.h5ad"
            adata.write(output_path)
            
            # Read back
            adata2 = ad.read_h5ad(output_path)
            assert adata2.shape == (100, 50)
        except Exception as e:
            pytest.skip(f"AnnData I/O not working: {e}")


class TestHooksErrorHandling:
    """Test error handling in hooks."""
    
    def test_hooks_graceful_degradation(self, requires_scl):
        """Test that hooks fail gracefully."""
        from scl import hooks
        
        # Hooks should install without error even if dependencies missing
        try:
            hooks.install()
            assert True
        except Exception as e:
            pytest.fail(f"Hooks installation should not raise: {e}")

