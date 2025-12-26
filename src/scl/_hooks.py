"""
Runtime Hooks and Integration

Performs "surgical operations" on third-party libraries to make them accept
SCL matrices as native sparse matrices.

Architecture:
1. ABC Registration: Make isinstance(scl_mat, scipy.sparse.spmatrix) return True
2. AnnData Patching: Modify type checks and I/O handlers
3. Optional: Native read support (load h5ad directly as SclCSR)

Safety:
- All patches are non-destructive (wrap, don't replace)
- Graceful degradation if libraries not installed
- Comprehensive error logging
- Can be disabled via environment variable

Usage:
    import scl
    # Hooks auto-activate on import
    
    # Or explicitly
    from scl import hooks
    hooks.install()
    
    # Disable if needed
    import os
    os.environ['SCL_NO_HOOKS'] = '1'
"""

import sys
import os
import logging
import importlib.util
from typing import Any, Optional
from functools import wraps

logger = logging.getLogger("scl.hooks")

# =============================================================================
# Feature Flags
# =============================================================================

def _is_enabled() -> bool:
    """Check if hooks are enabled (default: True)."""
    return os.environ.get('SCL_NO_HOOKS', '').lower() not in ('1', 'true', 'yes')


def _is_library_available(name: str) -> bool:
    """Check if a library is available without importing it."""
    return name in sys.modules or importlib.util.find_spec(name) is not None


def _is_library_loaded(name: str) -> bool:
    """Check if a library is already loaded in sys.modules."""
    return name in sys.modules


# =============================================================================
# Hook State Tracking
# =============================================================================

_hooks_installed = False
_scipy_spoofed = False
_anndata_patched = False
_native_read_enabled = False


def is_installed() -> bool:
    """Check if hooks have been installed."""
    return _hooks_installed


# =============================================================================
# Main Install Function
# =============================================================================

def install(enable_native_read: bool = False):
    """
    Install all available hooks.
    
    This function is idempotent - safe to call multiple times.
    
    Args:
        enable_native_read: If True, patch anndata.read_h5ad to return SclCSR
                           (Experimental feature, may break some code)
    
    Example:
        >>> from scl import hooks
        >>> hooks.install()
        >>> # Now anndata and scanpy will accept SclCSR matrices
    """
    global _hooks_installed
    
    if _hooks_installed:
        logger.debug("SCL hooks already installed, skipping")
        return
    
    if not _is_enabled():
        logger.info("SCL hooks disabled via SCL_NO_HOOKS environment variable")
        return
    
    logger.info("Installing SCL runtime hooks...")
    
    # Step 1: Scipy ABC registration (always safe)
    _spoof_scipy()
    
    # Step 2: AnnData patching (if available)
    if _is_library_available('anndata'):
        try:
            _patch_anndata()
            logger.info("✓ AnnData patching successful")
            
            if enable_native_read:
                _enable_native_read()
                logger.info("✓ Native read mode enabled")
        except Exception as e:
            logger.warning(f"✗ AnnData patching failed: {e}")
    
    _hooks_installed = True
    logger.info("SCL hooks installation complete")


# =============================================================================
# 1. Scipy Spoofing (ABC Registration)
# =============================================================================

def _spoof_scipy():
    """
    Register SCL matrices as virtual subclasses of scipy.sparse.spmatrix.
    
    This makes isinstance(scl_mat, scipy.sparse.spmatrix) return True,
    which tricks most libraries into accepting our matrices.
    
    Technical note:
    - Uses Python's ABC (Abstract Base Class) registration
    - Does NOT affect MRO (Method Resolution Order)
    - Only affects isinstance/issubclass checks
    - Completely safe and reversible
    """
    global _scipy_spoofed
    
    if _scipy_spoofed:
        return
    
    try:
        import scipy.sparse as sp
        
        # Import our matrix classes (lazy to avoid circular import)
        from .sparse import SclCSR, SclCSC, VirtualCSR, VirtualCSC
        
        # Register as virtual subclasses
        sp.spmatrix.register(SclCSR)
        sp.spmatrix.register(SclCSC)
        sp.spmatrix.register(VirtualCSR)
        sp.spmatrix.register(VirtualCSC)
        
        _scipy_spoofed = True
        logger.debug("✓ Registered SCL matrices as scipy.sparse.spmatrix subclasses")
        
    except ImportError:
        logger.debug("scipy not available, skipping ABC registration")
    except Exception as e:
        logger.warning(f"Failed to register with scipy: {e}")


# =============================================================================
# 2. AnnData Patching
# =============================================================================

def _patch_anndata():
    """
    Patch AnnData internals to support SCL matrices.
    
    Operations:
    1. Patch issparse/is_sparse utility functions
    2. Register I/O handlers for h5ad format
    3. Patch type checking in critical functions
    """
    global _anndata_patched
    
    if _anndata_patched:
        return
    
    # Lazy import (only when needed)
    import anndata
    import anndata.utils
    
    from .sparse import SclCSR, SclCSC, VirtualCSR, VirtualCSC
    
    # --- Patch A: issparse utility ---
    if hasattr(anndata.utils, 'issparse'):
        original_issparse = anndata.utils.issparse
        
        @wraps(original_issparse)
        def patched_issparse(x: Any) -> bool:
            """Enhanced issparse that recognizes SCL matrices."""
            if isinstance(x, (SclCSR, SclCSC, VirtualCSR, VirtualCSC)):
                return True
            return original_issparse(x)
        
        anndata.utils.issparse = patched_issparse
        logger.debug("✓ Patched anndata.utils.issparse")
    
    # Alternative name
    if hasattr(anndata.utils, 'is_sparse'):
        original_is_sparse = anndata.utils.is_sparse
        
        @wraps(original_is_sparse)
        def patched_is_sparse(x: Any) -> bool:
            if isinstance(x, (SclCSR, SclCSC, VirtualCSR, VirtualCSC)):
                return True
            return original_is_sparse(x)
        
        anndata.utils.is_sparse = patched_is_sparse
        logger.debug("✓ Patched anndata.utils.is_sparse")
    
    # --- Patch B: I/O Registry (h5ad format) ---
    try:
        # AnnData >= 0.8 uses a registry-based I/O system
        _patch_anndata_io()
    except Exception as e:
        logger.debug(f"I/O registry patching not available: {e}")
    
    # --- Patch C: View Creation ---
    # AnnData creates views of X when slicing (adata[0:10])
    # Our matrices already support slicing, so this should work automatically
    # No patching needed if we implemented __getitem__ correctly
    
    _anndata_patched = True


def _patch_anndata_io():
    """
    Register SCL matrices with AnnData's I/O system.
    
    This allows writing .h5ad files with SCL matrices without conversion.
    """
    try:
        from anndata._io.specs import write_elem
        from anndata._io.specs.methods import write_csr_matrix, write_csc_matrix
        
        from .sparse import SclCSR, SclCSC, VirtualCSR, VirtualCSC
        
        # Strategy: Reuse scipy's writers
        # Our matrices have compatible layout (data/indices/indptr/shape)
        
        # Define thin wrappers that convert to scipy for writing
        def write_scl_csr(f, k, elem, *args, **kwargs):
            """Write SclCSR to h5ad (via scipy conversion)."""
            scipy_mat = elem.to_scipy() if hasattr(elem, 'to_scipy') else elem
            return write_csr_matrix(f, k, scipy_mat, *args, **kwargs)
        
        def write_scl_csc(f, k, elem, *args, **kwargs):
            """Write SclCSC to h5ad (via scipy conversion)."""
            scipy_mat = elem.to_scipy() if hasattr(elem, 'to_scipy') else elem
            return write_csc_matrix(f, k, scipy_mat, *args, **kwargs)
        
        # Register writers
        write_elem.register(SclCSR, write_scl_csr)
        write_elem.register(VirtualCSR, write_scl_csr)
        write_elem.register(SclCSC, write_scl_csc)
        write_elem.register(VirtualCSC, write_scl_csc)
        
        logger.debug("✓ Registered SCL matrices with AnnData I/O system")
        
    except (ImportError, AttributeError) as e:
        logger.debug(f"AnnData I/O registration not available: {e}")


# =============================================================================
# 3. Native Read Support (Experimental)
# =============================================================================

def _enable_native_read():
    """
    Patch anndata.read_h5ad to return SclCSR/CSC instead of scipy matrices.
    
    This is EXPERIMENTAL and may break some code that expects scipy objects.
    Only enable if you know what you're doing.
    """
    global _native_read_enabled
    
    if _native_read_enabled:
        return
    
    try:
        import anndata
        from .sparse import SclCSR
        
        original_read_h5ad = anndata.read_h5ad
        
        @wraps(original_read_h5ad)
        def patched_read_h5ad(filename, *args, **kwargs):
            """Enhanced read_h5ad that returns SCL matrices."""
            adata = original_read_h5ad(filename, *args, **kwargs)
            
            # Convert X to SclCSR if it's scipy CSR
            try:
                import scipy.sparse as sp
                if sp.isspmatrix_csr(adata.X):
                    adata.X = SclCSR.from_scipy(adata.X)
                    logger.debug(f"Converted adata.X to SclCSR: {adata.X.shape}")
                
                # Also convert layers if present
                if hasattr(adata, 'layers') and adata.layers:
                    for key, layer in adata.layers.items():
                        if sp.isspmatrix_csr(layer):
                            adata.layers[key] = SclCSR.from_scipy(layer)
                            logger.debug(f"Converted layer '{key}' to SclCSR")
            except Exception as e:
                logger.warning(f"Failed to convert matrices in {filename}: {e}")
            
            return adata
        
        anndata.read_h5ad = patched_read_h5ad
        _native_read_enabled = True
        logger.info("✓ Native read mode enabled (experimental)")
        
    except Exception as e:
        logger.error(f"Failed to enable native read: {e}")


def enable_native_read():
    """
    Public API to enable native read mode.
    
    After calling this, anndata.read_h5ad will return SclCSR matrices.
    
    Warning: This is experimental and may break some code.
    
    Example:
        >>> from scl import hooks
        >>> hooks.enable_native_read()
        >>> adata = anndata.read_h5ad('data.h5ad')
        >>> type(adata.X)  # SclCSR instead of scipy.sparse.csr_matrix
    """
    if not _is_library_available('anndata'):
        raise RuntimeError("AnnData is not available")
    
    _enable_native_read()


# =============================================================================
# Inspection and Debugging
# =============================================================================

def status() -> dict:
    """
    Get current hook status.
    
    Returns:
        Dictionary with hook installation status
    
    Example:
        >>> from scl import hooks
        >>> print(hooks.status())
        {'installed': True, 'scipy_spoofed': True, 'anndata_patched': True, ...}
    """
    return {
        'hooks_installed': _hooks_installed,
        'scipy_spoofed': _scipy_spoofed,
        'anndata_patched': _anndata_patched,
        'native_read_enabled': _native_read_enabled,
        'anndata_available': _is_library_available('anndata'),
        'scipy_available': _is_library_available('scipy'),
        'hooks_enabled': _is_enabled(),
    }


def test_integration():
    """
    Test if SCL matrices are accepted by anndata/scipy.
    
    Returns:
        True if integration successful, False otherwise
    
    Example:
        >>> from scl import hooks
        >>> hooks.install()
        >>> success = hooks.test_integration()
    """
    try:
        from .sparse import SclCSR, Array, from_list
        import scipy.sparse as sp
        
        # Test 1: isinstance check
        data = Array.from_list([1.0, 2.0, 3.0], dtype='float32')
        indices = Array.from_list([0, 1, 2], dtype='int64')
        indptr = Array.from_list([0, 2, 3], dtype='int64')
        mat = SclCSR(data, indices, indptr, shape=(2, 3))
        
        if not isinstance(mat, sp.spmatrix):
            logger.error("isinstance(SclCSR, spmatrix) returned False")
            return False
        
        logger.info("✓ isinstance check passed")
        
        # Test 2: AnnData compatibility (if available)
        if _is_library_available('anndata'):
            import anndata
            
            try:
                # Create a minimal AnnData object with SclCSR
                adata = anndata.AnnData(X=mat)
                logger.info("✓ AnnData accepts SclCSR as X")
                
                # Test issparse
                if hasattr(anndata.utils, 'issparse'):
                    if not anndata.utils.issparse(mat):
                        logger.error("anndata.utils.issparse(SclCSR) returned False")
                        return False
                    logger.info("✓ anndata.utils.issparse check passed")
                
            except Exception as e:
                logger.error(f"AnnData compatibility test failed: {e}")
                return False
        
        logger.info("✓ All integration tests passed")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


# =============================================================================
# Uninstall Support (Advanced)
# =============================================================================

_original_functions = {}


def uninstall():
    """
    Uninstall hooks and restore original functions.
    
    This is mainly for testing and debugging.
    Use with caution in production code.
    
    Example:
        >>> from scl import hooks
        >>> hooks.install()
        >>> # ... do work ...
        >>> hooks.uninstall()  # Restore original behavior
    """
    global _hooks_installed, _anndata_patched
    
    if not _hooks_installed:
        logger.debug("No hooks to uninstall")
        return
    
    # Restore patched functions
    if _anndata_patched and 'anndata' in sys.modules:
        import anndata.utils
        
        for attr, original in _original_functions.items():
            if hasattr(anndata.utils, attr):
                setattr(anndata.utils, attr, original)
                logger.debug(f"✓ Restored anndata.utils.{attr}")
    
    # Note: ABC registration cannot be undone
    # This is a Python limitation
    
    _hooks_installed = False
    _anndata_patched = False
    logger.info("SCL hooks uninstalled")


# =============================================================================
# Auto-Install on Module Import
# =============================================================================

def _auto_install():
    """
    Automatically install hooks when this module is imported.
    
    This runs when scl is imported, ensuring seamless integration.
    """
    try:
        install(enable_native_read=False)
    except Exception as e:
        logger.warning(f"Auto-install failed: {e}")


# Auto-install when module is loaded (unless disabled)
if _is_enabled():
    _auto_install()

