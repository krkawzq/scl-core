"""Data transformation operations.

High-level API for log transforms, softmax, and other data transformations.
"""

from typing import Union, TYPE_CHECKING, overload

if TYPE_CHECKING:
    from scl.sparse import SclCSR, SclCSC, Array


__all__ = ['log1p', 'softmax']


# =============================================================================
# Log Transforms
# =============================================================================

def log1p(mat: "SclCSR", inplace: bool = False) -> "SclCSR":
    """Apply natural log(1 + x) transformation to matrix values.
    
    Args:
        mat: Input sparse matrix (CSR format).
        inplace: If True, modify matrix in place. If False, return a new matrix.
        
    Returns:
        Transformed matrix.
        
    Example:
        >>> from scl.sparse import SclCSR
        >>> mat = SclCSR.from_dense([[1, 2], [3, 4]])
        >>> transformed = log1p(mat)
    """
    from scl.sparse import SclCSR
    from scl._typing import get_format, ensure_scl_csr
    
    fmt = get_format(mat)
    
    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np
        result = mat.copy()
        result.data = np.log1p(result.data)
        return result
    
    # Native SCL
    csr = ensure_scl_csr(mat)
    csr.materialize()
    
    if not inplace:
        csr = csr.copy()
    
    # Try to use C++ kernel
    try:
        from scl._kernel import transform as kernel_transform
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        data_ptr = csr.data.get_pointer()
        indices_ptr = csr.indices.get_pointer()
        indptr_ptr = csr.indptr.get_pointer()
        
        kernel_transform.log1p_inplace_csr(
            data_ptr, indices_ptr, indptr_ptr,
            csr.shape[0], csr.shape[1]
        )
        
        return csr
        
    except (ImportError, LibraryNotFoundError, RuntimeError, AttributeError):
        # Fallback to Python implementation
        import math
        for k in range(csr.nnz):
            csr._data[k] = math.log1p(csr._data[k])
        return csr


# =============================================================================
# Softmax
# =============================================================================

def softmax(mat: "SclCSR", axis: int = 1, inplace: bool = False) -> "SclCSR":
    """Apply softmax transformation along specified axis.
    
    Softmax normalizes values so they sum to 1 and represent probabilities.
    For sparse matrices, only non-zero elements are considered.
    
    Args:
        mat: Input sparse matrix (CSR format).
        axis: Axis along which to apply softmax.
            - axis=1: Apply softmax per row (default)
            - axis=0: Apply softmax per column (converts to CSC internally)
        inplace: If True, modify matrix in place. If False, return a new matrix.
        
    Returns:
        Transformed matrix with softmax applied.
        
    Example:
        >>> from scl.sparse import SclCSR
        >>> mat = SclCSR.from_dense([[1, 2], [3, 4]])
        >>> probs = softmax(mat, axis=1)  # Row-wise softmax
        
    Notes:
        - Softmax is computed as: exp(x_i) / sum(exp(x_j))
        - For numerical stability, max value is subtracted before exp
        - Only non-zero elements participate in softmax computation
    """
    from scl.sparse import SclCSR
    from scl._typing import get_format, ensure_scl_csr
    
    fmt = get_format(mat)
    
    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np
        from scipy import sparse
        
        result = mat.tocsr().copy() if not inplace else mat.tocsr()
        
        if axis == 1:  # Row-wise
            for i in range(result.shape[0]):
                start = result.indptr[i]
                end = result.indptr[i + 1]
                if start < end:
                    row_data = result.data[start:end]
                    row_data -= row_data.max()  # Numerical stability
                    exp_data = np.exp(row_data)
                    result.data[start:end] = exp_data / exp_data.sum()
        else:  # Column-wise
            csc = result.tocsc()
            for j in range(csc.shape[1]):
                start = csc.indptr[j]
                end = csc.indptr[j + 1]
                if start < end:
                    col_data = csc.data[start:end]
                    col_data -= col_data.max()
                    exp_data = np.exp(col_data)
                    csc.data[start:end] = exp_data / exp_data.sum()
            result = csc.tocsr()
        
        return result
    
    # Native SCL
    csr = ensure_scl_csr(mat)
    csr.materialize()
    
    if not inplace:
        csr = csr.copy()
    
    if axis == 1:  # Row-wise softmax
        # Try to use C++ kernel
        try:
            from scl._kernel import transform as kernel_transform
            from scl._kernel.lib_loader import LibraryNotFoundError
            
            data_ptr = csr.data.get_pointer()
            indices_ptr = csr.indices.get_pointer()
            indptr_ptr = csr.indptr.get_pointer()
            
            kernel_transform.softmax_inplace_csr(
                data_ptr, indices_ptr, indptr_ptr,
                csr.shape[0], csr.shape[1]
            )
            
            return csr
            
        except (ImportError, LibraryNotFoundError, RuntimeError, AttributeError):
            pass  # Fall through to Python implementation
        
        # Fallback to Python implementation
        import math
        for i in range(csr.shape[0]):
            start = csr._indptr[i]
            end = csr._indptr[i + 1]
            if start < end:
                # Find max for numerical stability
                max_val = csr._data[start]
                for k in range(start + 1, end):
                    if csr._data[k] > max_val:
                        max_val = csr._data[k]
                
                # Compute exp and sum
                exp_sum = 0.0
                for k in range(start, end):
                    csr._data[k] = math.exp(csr._data[k] - max_val)
                    exp_sum += csr._data[k]
                
                # Normalize
                for k in range(start, end):
                    csr._data[k] /= exp_sum
        
        return csr
    
    else:  # Column-wise softmax (axis=0)
        # Convert to CSC, apply, convert back
        csc = csr.to_csc()
        
        import math
        for j in range(csc.shape[1]):
            start = csc._indptr[j]
            end = csc._indptr[j + 1]
            if start < end:
                # Find max for numerical stability
                max_val = csc._data[start]
                for k in range(start + 1, end):
                    if csc._data[k] > max_val:
                        max_val = csc._data[k]
                
                # Compute exp and sum
                exp_sum = 0.0
                for k in range(start, end):
                    csc._data[k] = math.exp(csc._data[k] - max_val)
                    exp_sum += csc._data[k]
                
                # Normalize
                for k in range(start, end):
                    csc._data[k] /= exp_sum
        
        return csc.to_csr()

