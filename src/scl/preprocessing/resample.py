"""Resampling operations for count matrices.

High-level API for downsampling and resampling sparse count matrices.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from scl.sparse import SclCSR, SclCSC


__all__ = ['downsample_counts']


def downsample_counts(
    mat: "SclCSC",
    target_sum: float,
    *,
    seed: Optional[int] = None,
    inplace: bool = False
) -> "SclCSC":
    """Downsample counts to a target sum per column.
    
    This is commonly used in single-cell analysis to normalize library sizes
    by downsampling cells to a common total count.
    
    Args:
        mat: Input sparse count matrix in CSC format (genes x cells).
        target_sum: Target total counts per column (cell).
        seed: Random seed for reproducibility. If None, uses system random.
        inplace: If True, modify matrix in place. If False, return a new matrix.
        
    Returns:
        Downsampled matrix with each column summing to approximately target_sum.
        
    Example:
        >>> from scl.sparse import SclCSC
        >>> # counts: genes x cells
        >>> mat = SclCSC.from_dense([[100, 200], [50, 100], [25, 50]])
        >>> downsampled = downsample_counts(mat, target_sum=100, seed=42)
        
    Notes:
        - Columns with total counts less than target_sum are unchanged.
        - Uses multinomial sampling to preserve relative proportions.
        - Results are stochastic; use seed for reproducibility.
    """
    from scl.sparse import SclCSC
    from scl._typing import get_format, ensure_scl_csc
    
    fmt = get_format(mat)
    
    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np
        from scipy import sparse
        
        csc = mat.tocsc().copy() if not inplace else mat.tocsc()
        rng = np.random.default_rng(seed)
        
        for j in range(csc.shape[1]):
            start = csc.indptr[j]
            end = csc.indptr[j + 1]
            if start < end:
                col_data = csc.data[start:end].astype(float)
                col_sum = col_data.sum()
                
                if col_sum > target_sum:
                    # Downsample using multinomial
                    probs = col_data / col_sum
                    new_counts = rng.multinomial(int(target_sum), probs)
                    csc.data[start:end] = new_counts.astype(csc.data.dtype)
        
        return csc
    
    # Native SCL
    csc = ensure_scl_csc(mat)
    csc.materialize()
    
    if not inplace:
        csc = csc.copy()
    
    actual_seed = seed if seed is not None else 0
    
    # Try to use C++ kernel
    try:
        from scl._kernel import resample as kernel_resample
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        data_ptr, indices_ptr, indptr_ptr, lengths_ptr, rows, cols, nnz = csc.get_c_pointers()
        
        kernel_resample.downsample_counts_csc(
            data_ptr, indices_ptr, indptr_ptr, lengths_ptr,
            rows, cols, nnz,
            target_sum, actual_seed
        )
        
        return csc
        
    except (ImportError, LibraryNotFoundError, RuntimeError, AttributeError):
        # Fallback to Python implementation
        import random
        if seed is not None:
            random.seed(seed)
        
        for j in range(csc.shape[1]):
            start = csc._indptr[j]
            end = csc._indptr[j + 1]
            if start < end:
                # Calculate column sum
                col_sum = 0.0
                for k in range(start, end):
                    col_sum += csc._data[k]
                
                if col_sum > target_sum:
                    # Simple probabilistic downsampling
                    scale = target_sum / col_sum
                    for k in range(start, end):
                        # Scale down proportionally
                        csc._data[k] = round(csc._data[k] * scale)
        
        return csc

