"""High-Level Sparse Matrix Operations.

This module provides functional operations on sparse matrices:
- Stacking operations (vstack, hstack)
- Format conversions
- Cross-platform conversions (scipy, anndata, numpy)
- Matrix alignment operations
- Statistical operations

All operations work transparently with any backend (Custom, Virtual, Mapped)
and automatically maintain ownership and reference chains.

Example:
    >>> from scl.sparse import SclCSR, vstack_csr, hstack_csc
    >>> 
    >>> # Stack matrices
    >>> stacked = vstack_csr([mat1, mat2, mat3])
    >>> 
    >>> # Convert formats
    >>> csc = mat.tocsc()
    >>> 
    >>> # Cross-platform
    >>> scipy_mat = mat.to_scipy()
    >>> anndata_x = mat.to_anndata()
"""

from typing import List, Union, Optional, Any, Tuple
import warnings

from ._array import Array, zeros, empty, from_list
from ._backend import Backend, Ownership, VirtualStorage, ChunkInfo
from ._ownership import RefChain
from ._csr import SclCSR
from ._csc import SclCSC

__all__ = [
    # Stacking
    'vstack_csr',
    'hstack_csc',
    'vstack',
    'hstack',
    
    # Conversions
    'convert_format',
    
    # Cross-platform
    'from_scipy',
    'from_anndata',
    'from_numpy',
    'to_scipy',
    'to_anndata',
    'to_numpy',
    
    # Alignment
    'align_rows',
    'align_cols',
    'align_to_categories',
    
    # Statistical
    'sum_rows',
    'sum_cols',
    'mean_rows',
    'mean_cols',
    'var_rows',
    'var_cols',
]


# =============================================================================
# Stacking Operations
# =============================================================================

def vstack_csr(matrices: List['SclCSR']) -> 'SclCSR':
    """Vertically stack CSR matrices (row concatenation).
    
    Creates a Virtual backend matrix without copying data.
    The source matrices are kept alive via reference chain.
    
    Args:
        matrices: List of SclCSR matrices with same column count.
        
    Returns:
        SclCSR with Virtual backend representing stacked matrices.
        
    Example:
        >>> mat1 = SclCSR.from_dense([[1, 2], [3, 4]])  # 2x2
        >>> mat2 = SclCSR.from_dense([[5, 6]])          # 1x2
        >>> stacked = vstack_csr([mat1, mat2])          # 3x2
        >>> print(stacked.shape)  # (3, 2)
        >>> print(stacked.backend)  # Backend.VIRTUAL
    """
    from ._csr import SclCSR
    
    if len(matrices) == 0:
        return SclCSR.zeros(0, 0)
    
    if len(matrices) == 1:
        return matrices[0]
    
    return SclCSR._create_virtual(matrices)


def hstack_csc(matrices: List['SclCSC']) -> 'SclCSC':
    """Horizontally stack CSC matrices (column concatenation).
    
    Creates a Virtual backend matrix without copying data.
    
    Args:
        matrices: List of SclCSC matrices with same row count.
        
    Returns:
        SclCSC with Virtual backend.
        
    Example:
        >>> mat1 = SclCSC.from_dense([[1, 2], [3, 4]])  # 2x2
        >>> mat2 = SclCSC.from_dense([[5], [6]])        # 2x1
        >>> stacked = hstack_csc([mat1, mat2])          # 2x3
    """
    from ._csc import SclCSC
    
    if len(matrices) == 0:
        return SclCSC.zeros(0, 0)
    
    if len(matrices) == 1:
        return matrices[0]
    
    return SclCSC._create_virtual(matrices)


def vstack(matrices: List[Union['SclCSR', 'SclCSC']]) -> Union['SclCSR', 'SclCSC']:
    """Vertically stack matrices (auto-detect format).
    
    Converts CSC to CSR if needed for optimal vstack.
    
    Args:
        matrices: List of sparse matrices.
        
    Returns:
        Stacked matrix (CSR format).
    """
    from ._csr import SclCSR
    from ._csc import SclCSC
    
    csr_matrices = []
    for mat in matrices:
        if isinstance(mat, SclCSC):
            csr_matrices.append(mat.tocsr())
        elif isinstance(mat, SclCSR):
            csr_matrices.append(mat)
        else:
            raise TypeError(f"Expected SclCSR or SclCSC, got {type(mat)}")
    
    return vstack_csr(csr_matrices)


def hstack(matrices: List[Union['SclCSR', 'SclCSC']]) -> Union['SclCSR', 'SclCSC']:
    """Horizontally stack matrices (auto-detect format).
    
    Converts CSR to CSC if needed for optimal hstack.
    
    Args:
        matrices: List of sparse matrices.
        
    Returns:
        Stacked matrix (CSC format).
    """
    from ._csr import SclCSR
    from ._csc import SclCSC
    
    csc_matrices = []
    for mat in matrices:
        if isinstance(mat, SclCSR):
            csc_matrices.append(mat.tocsc())
        elif isinstance(mat, SclCSC):
            csc_matrices.append(mat)
        else:
            raise TypeError(f"Expected SclCSR or SclCSC, got {type(mat)}")
    
    return hstack_csc(csc_matrices)


# =============================================================================
# Format Conversion
# =============================================================================

def convert_format(
    mat: Union['SclCSR', 'SclCSC'],
    target: str
) -> Union['SclCSR', 'SclCSC']:
    """Convert between CSR and CSC formats.
    
    Args:
        mat: Source matrix.
        target: Target format ('csr' or 'csc').
        
    Returns:
        Converted matrix.
        
    Example:
        >>> csr = SclCSR.from_dense([[1, 2]])
        >>> csc = convert_format(csr, 'csc')
    """
    from ._csr import SclCSR
    from ._csc import SclCSC
    
    target = target.lower()
    
    if target == 'csr':
        if isinstance(mat, SclCSR):
            return mat
        return mat.tocsr()
    elif target == 'csc':
        if isinstance(mat, SclCSC):
            return mat
        return mat.tocsc()
    else:
        raise ValueError(f"Unknown format: {target}. Use 'csr' or 'csc'")


# =============================================================================
# Cross-Platform Conversions
# =============================================================================

def from_scipy(mat: Any, copy: bool = False) -> Union['SclCSR', 'SclCSC']:
    """Create SCL matrix from scipy.sparse matrix.
    
    Auto-detects format (CSR/CSC) and creates appropriate type.
    
    Args:
        mat: scipy sparse matrix.
        copy: If True, copy data instead of borrowing.
        
    Returns:
        SclCSR or SclCSC.
        
    Example:
        >>> import scipy.sparse as sp
        >>> scipy_mat = sp.csr_matrix([[1, 2], [3, 4]])
        >>> scl_mat = from_scipy(scipy_mat)
    """
    from ._csr import SclCSR
    from ._csc import SclCSC
    
    try:
        import scipy.sparse as sp
    except ImportError:
        raise ImportError("scipy required for from_scipy()")
    
    if sp.isspmatrix_csr(mat):
        return SclCSR.from_scipy(mat, copy=copy)
    elif sp.isspmatrix_csc(mat):
        return SclCSC.from_scipy(mat, copy=copy)
    else:
        # Convert to CSR
        return SclCSR.from_scipy(mat.tocsr(), copy=True)


def from_anndata(
    adata: Any,
    layer: Optional[str] = None,
    copy: bool = False
) -> Union['SclCSR', 'SclCSC']:
    """Create SCL matrix from AnnData X or layer.
    
    Args:
        adata: AnnData object.
        layer: Optional layer name. If None, uses X.
        copy: If True, copy data.
        
    Returns:
        SclCSR or SclCSC.
        
    Example:
        >>> import anndata
        >>> adata = anndata.read_h5ad("data.h5ad")
        >>> mat = from_anndata(adata)
    """
    from ._csr import SclCSR
    from ._csc import SclCSC
    
    try:
        import scipy.sparse as sp
        import numpy as np
    except ImportError:
        raise ImportError("scipy and numpy required for from_anndata()")
    
    if layer is not None:
        data = adata.layers[layer]
    else:
        data = adata.X
    
    if sp.issparse(data):
        return from_scipy(data, copy=copy)
    elif isinstance(data, np.ndarray):
        return from_numpy(data)
    else:
        raise TypeError(f"Unknown data type in AnnData: {type(data)}")


def from_numpy(
    arr: Any,
    dtype: str = 'float64'
) -> 'SclCSR':
    """Create CSR matrix from numpy 2D array.
    
    Args:
        arr: 2D numpy array.
        dtype: Target dtype.
        
    Returns:
        SclCSR matrix.
    """
    from ._csr import SclCSR
    
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy required for from_numpy()")
    
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")
    
    # Convert to list and use from_dense
    return SclCSR.from_dense(arr.tolist(), dtype=dtype)


def to_scipy(mat: Union['SclCSR', 'SclCSC']) -> Any:
    """Convert SCL matrix to scipy.sparse matrix.
    
    Args:
        mat: SCL sparse matrix.
        
    Returns:
        scipy.sparse.csr_matrix or csc_matrix.
    """
    return mat.to_scipy()


def to_anndata(
    mat: Union['SclCSR', 'SclCSC'],
    obs: Optional[Any] = None,
    var: Optional[Any] = None
) -> Any:
    """Create AnnData object from SCL matrix.
    
    Args:
        mat: SCL sparse matrix.
        obs: Optional observation annotations.
        var: Optional variable annotations.
        
    Returns:
        AnnData object.
        
    Example:
        >>> adata = to_anndata(mat)
        >>> adata.write_h5ad("output.h5ad")
    """
    try:
        import anndata
        import pandas as pd
    except ImportError:
        raise ImportError("anndata and pandas required for to_anndata()")
    
    scipy_mat = mat.to_scipy()
    
    if obs is None:
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(mat.rows)])
    if var is None:
        var = pd.DataFrame(index=[f"gene_{j}" for j in range(mat.cols)])
    
    return anndata.AnnData(X=scipy_mat, obs=obs, var=var)


def to_numpy(mat: Union['SclCSR', 'SclCSC']) -> Any:
    """Convert SCL matrix to dense numpy array.
    
    Warning:
        This can use a lot of memory for large matrices!
        
    Args:
        mat: SCL sparse matrix.
        
    Returns:
        2D numpy array.
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy required for to_numpy()")
    
    scipy_mat = mat.to_scipy()
    return scipy_mat.toarray()


# =============================================================================
# Alignment Operations
# =============================================================================

def align_rows(
    mat: 'SclCSR',
    mapping: Union[List[int], Array],
    new_rows: int
) -> 'SclCSR':
    """Align matrix rows according to mapping.
    
    Reorders and/or subsets rows based on a mapping array.
    Negative values in mapping indicate rows to skip.
    
    Args:
        mat: Source CSR matrix.
        mapping: For each new row, the index in source (-1 to skip).
        new_rows: Number of rows in output.
        
    Returns:
        Aligned SclCSR matrix.
        
    Example:
        >>> # Reorder rows: [row2, row0, row1]
        >>> aligned = align_rows(mat, [2, 0, 1], 3)
        
        >>> # Subset: keep rows 0 and 2
        >>> aligned = align_rows(mat, [0, 2], 2)
        
        >>> # Insert empty row
        >>> aligned = align_rows(mat, [0, -1, 1], 3)
    """
    from ._csr import SclCSR
    
    if isinstance(mapping, list):
        mapping = from_list(mapping, dtype='int64')
    
    mat._ensure_custom()
    
    # Count nnz
    total_nnz = 0
    for i in range(new_rows):
        src_idx = mapping[i]
        if src_idx >= 0:
            total_nnz += mat._storage.primary_lengths[src_idx]
    
    # Allocate
    new_data = empty(total_nnz, dtype=mat.dtype)
    new_indices = empty(total_nnz, dtype='int64')
    new_indptr = zeros(new_rows + 1, dtype='int64')
    
    # Copy
    pos = 0
    for i in range(new_rows):
        src_idx = mapping[i]
        if src_idx >= 0:
            start = mat._storage.indptr[src_idx]
            end = mat._storage.indptr[src_idx + 1]
            
            for k in range(start, end):
                new_data[pos] = mat._storage.data[k]
                new_indices[pos] = mat._storage.indices[k]
                pos += 1
        
        new_indptr[i + 1] = pos
    
    return SclCSR(new_data, new_indices, new_indptr, shape=(new_rows, mat.cols), ownership=Ownership.OWNED)


def align_cols(
    mat: 'SclCSC',
    mapping: Union[List[int], Array],
    new_cols: int
) -> 'SclCSC':
    """Align matrix columns according to mapping.
    
    Similar to align_rows but for CSC matrices.
    
    Args:
        mat: Source CSC matrix.
        mapping: For each new col, the index in source (-1 to skip).
        new_cols: Number of columns in output.
        
    Returns:
        Aligned SclCSC matrix.
    """
    from ._csc import SclCSC
    
    if isinstance(mapping, list):
        mapping = from_list(mapping, dtype='int64')
    
    mat._ensure_custom()
    
    # Count nnz
    total_nnz = 0
    for j in range(new_cols):
        src_idx = mapping[j]
        if src_idx >= 0:
            total_nnz += mat._storage.primary_lengths[src_idx]
    
    # Allocate
    new_data = empty(total_nnz, dtype=mat.dtype)
    new_indices = empty(total_nnz, dtype='int64')
    new_indptr = zeros(new_cols + 1, dtype='int64')
    
    # Copy
    pos = 0
    for j in range(new_cols):
        src_idx = mapping[j]
        if src_idx >= 0:
            start = mat._storage.indptr[src_idx]
            end = mat._storage.indptr[src_idx + 1]
            
            for k in range(start, end):
                new_data[pos] = mat._storage.data[k]
                new_indices[pos] = mat._storage.indices[k]
                pos += 1
        
        new_indptr[j + 1] = pos
    
    return SclCSC(new_data, new_indices, new_indptr, shape=(mat.rows, new_cols), ownership=Ownership.OWNED)


def align_to_categories(
    mat: Union['SclCSR', 'SclCSC'],
    source_categories: List[str],
    target_categories: List[str],
    axis: int = 0,
    fill_value: float = 0.0
) -> Union['SclCSR', 'SclCSC']:
    """Align matrix to new category order.
    
    Useful for aligning AnnData obs/var names.
    
    Args:
        mat: Source matrix.
        source_categories: Current category names.
        target_categories: Desired category names.
        axis: 0 for rows, 1 for columns.
        fill_value: Value for missing categories (ignored for sparse).
        
    Returns:
        Aligned matrix.
        
    Example:
        >>> # Align genes to reference
        >>> aligned = align_to_categories(
        ...     mat, 
        ...     source_categories=adata.var_names.tolist(),
        ...     target_categories=reference_genes,
        ...     axis=1
        ... )
    """
    # Build mapping
    source_idx = {cat: i for i, cat in enumerate(source_categories)}
    
    mapping = []
    for cat in target_categories:
        if cat in source_idx:
            mapping.append(source_idx[cat])
        else:
            mapping.append(-1)  # Not found
    
    new_size = len(target_categories)
    
    if axis == 0:
        from ._csr import SclCSR
        if not isinstance(mat, SclCSR):
            mat = mat.tocsr()
        return align_rows(mat, mapping, new_size)
    else:
        from ._csc import SclCSC
        if not isinstance(mat, SclCSC):
            mat = mat.tocsc()
        return align_cols(mat, mapping, new_size)


# =============================================================================
# Statistical Operations
# =============================================================================

def sum_rows(mat: 'SclCSR') -> Array:
    """Compute row sums.
    
    Args:
        mat: CSR matrix.
        
    Returns:
        Array of row sums (length = rows).
    """
    mat._ensure_custom()
    
    result = zeros(mat.rows, dtype=mat.dtype)
    for i in range(mat.rows):
        start = mat._storage.indptr[i]
        end = mat._storage.indptr[i + 1]
        total = 0.0
        for k in range(start, end):
            total += mat._storage.data[k]
        result[i] = total
    
    return result


def sum_cols(mat: 'SclCSC') -> Array:
    """Compute column sums.
    
    Args:
        mat: CSC matrix.
        
    Returns:
        Array of column sums (length = cols).
    """
    mat._ensure_custom()
    
    result = zeros(mat.cols, dtype=mat.dtype)
    for j in range(mat.cols):
        start = mat._storage.indptr[j]
        end = mat._storage.indptr[j + 1]
        total = 0.0
        for k in range(start, end):
            total += mat._storage.data[k]
        result[j] = total
    
    return result


def mean_rows(mat: 'SclCSR') -> Array:
    """Compute row means.
    
    Args:
        mat: CSR matrix.
        
    Returns:
        Array of row means (length = rows).
    """
    sums = sum_rows(mat)
    
    result = zeros(mat.rows, dtype=mat.dtype)
    cols = float(mat.cols)
    for i in range(mat.rows):
        result[i] = sums[i] / cols
    
    return result


def mean_cols(mat: 'SclCSC') -> Array:
    """Compute column means.
    
    Args:
        mat: CSC matrix.
        
    Returns:
        Array of column means (length = cols).
    """
    sums = sum_cols(mat)
    
    result = zeros(mat.cols, dtype=mat.dtype)
    rows = float(mat.rows)
    for j in range(mat.cols):
        result[j] = sums[j] / rows
    
    return result


def var_rows(mat: 'SclCSR', ddof: int = 0) -> Array:
    """Compute row variances.
    
    Args:
        mat: CSR matrix.
        ddof: Delta degrees of freedom.
        
    Returns:
        Array of row variances.
    """
    mat._ensure_custom()
    
    means = mean_rows(mat)
    result = zeros(mat.rows, dtype=mat.dtype)
    n = float(mat.cols - ddof)
    
    for i in range(mat.rows):
        start = mat._storage.indptr[i]
        end = mat._storage.indptr[i + 1]
        mean = means[i]
        
        # Sum squared deviations from non-zeros
        sq_sum = 0.0
        for k in range(start, end):
            diff = mat._storage.data[k] - mean
            sq_sum += diff * diff
        
        # Add squared deviations from zeros
        nnz_row = end - start
        n_zeros = mat.cols - nnz_row
        sq_sum += n_zeros * mean * mean
        
        result[i] = sq_sum / n
    
    return result


def var_cols(mat: 'SclCSC', ddof: int = 0) -> Array:
    """Compute column variances.
    
    Args:
        mat: CSC matrix.
        ddof: Delta degrees of freedom.
        
    Returns:
        Array of column variances.
    """
    mat._ensure_custom()
    
    means = mean_cols(mat)
    result = zeros(mat.cols, dtype=mat.dtype)
    n = float(mat.rows - ddof)
    
    for j in range(mat.cols):
        start = mat._storage.indptr[j]
        end = mat._storage.indptr[j + 1]
        mean = means[j]
        
        sq_sum = 0.0
        for k in range(start, end):
            diff = mat._storage.data[k] - mean
            sq_sum += diff * diff
        
        nnz_col = end - start
        n_zeros = mat.rows - nnz_col
        sq_sum += n_zeros * mean * mean
        
        result[j] = sq_sum / n
    
    return result


# =============================================================================
# Additional Utilities
# =============================================================================

def concatenate(
    matrices: List[Union['SclCSR', 'SclCSC']],
    axis: int = 0
) -> Union['SclCSR', 'SclCSC']:
    """Concatenate matrices along axis.
    
    Args:
        matrices: List of matrices.
        axis: 0 for vstack, 1 for hstack.
        
    Returns:
        Concatenated matrix.
    """
    if axis == 0:
        return vstack(matrices)
    else:
        return hstack(matrices)


def empty_like(
    mat: Union['SclCSR', 'SclCSC'],
    nnz: Optional[int] = None
) -> Union['SclCSR', 'SclCSC']:
    """Create empty matrix with same shape/dtype.
    
    Args:
        mat: Template matrix.
        nnz: Number of non-zeros to allocate.
        
    Returns:
        Empty matrix.
    """
    from ._csr import SclCSR
    from ._csc import SclCSC
    
    if nnz is None:
        nnz = mat.nnz
    
    if isinstance(mat, SclCSR):
        return SclCSR.empty(mat.rows, mat.cols, nnz, mat.dtype)
    else:
        return SclCSC.empty(mat.rows, mat.cols, nnz, mat.dtype)


def zeros_like(
    mat: Union['SclCSR', 'SclCSC']
) -> Union['SclCSR', 'SclCSC']:
    """Create zero matrix with same shape/dtype.
    
    Args:
        mat: Template matrix.
        
    Returns:
        Zero matrix.
    """
    from ._csr import SclCSR
    from ._csc import SclCSC
    
    if isinstance(mat, SclCSR):
        return SclCSR.zeros(mat.rows, mat.cols, mat.dtype)
    else:
        return SclCSC.zeros(mat.rows, mat.cols, mat.dtype)
