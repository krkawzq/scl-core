"""
Virtual Matrix Wrappers

Zero-copy, composite wrappers for sparse matrices.
Supports logical stacking (vstack) and slicing without immediate memory allocation.

Design Philosophy:
1. Reference Chain: Strong refs to prevent GC
2. View Flattening: Avoid nested views (always flatten to base matrices)
3. Lazy Evaluation: Defer memory operations until materialization
4. Single/Multi Block: Optimize for common single-source case
"""

from typing import Tuple, Any, Optional, Union, List
from functools import wraps
import ctypes

from ._array import Array, zeros, empty, from_list
from .._kernel import utils as kernel_utils

from ._matrix import SclCSR, SclCSC

__all__ = ['VirtualCSR', 'VirtualCSC']


# =============================================================================
# Helper Structures (Private)
# =============================================================================

class _MatrixChunk:
    """
    Container for a matrix chunk with optional row/col mapping.
    
    Attributes:
        matrix: Backing matrix (Scipy or SclMatrix)
        row_map: Optional row indirection [None = identity]
        shape: Effective shape after applying map
        _cached_pointers: Cached C pointers for performance
    """
    
    __slots__ = ('matrix', 'row_map', 'shape', 'dtype', '_cached_pointers', '_row_lengths')
    
    def __init__(self, matrix: Any, row_map: Optional[Array] = None):
        """
        Initialize chunk.
        
        Args:
            matrix: Backing matrix (must have data/indices/indptr)
            row_map: Optional row indirection array
        """
        self.matrix = matrix  # Strong reference
        self.row_map = row_map
        
        # Determine dtype
        if hasattr(matrix, 'dtype'):
            self.dtype = matrix.dtype
        elif hasattr(matrix, 'data'):
            import numpy as np
            self.dtype = 'float32' if matrix.data.dtype == np.float32 else 'float64'
        else:
            raise TypeError("Cannot determine dtype from matrix")
        
        # Calculate effective shape
        if row_map is None:
            self.shape = matrix.shape
        else:
            self.shape = (len(row_map), matrix.shape[1])
        
        self._cached_pointers = None
        self._row_lengths = None
    
    def get_pointers(self) -> Tuple:
        """
        Get C pointers from backing matrix.
        
        Returns:
            (data_ptr, indices_ptr, indptr_ptr, row_lengths_ptr or None)
        """
        if self._cached_pointers is not None:
            return self._cached_pointers
        
        mat = self.matrix
        
        # Check if this is a Scipy matrix (has _scl_view cached arrays)
        if hasattr(mat, '_scl_view'):
            ptrs = (
                mat._scl_view_data.get_pointer(),
                mat._scl_view_indices.get_pointer(),
                mat._scl_view_indptr.get_pointer(),
                self.get_row_lengths().get_pointer() if self._row_lengths else None
            )
        elif hasattr(mat, 'get_c_pointers'):
            # SclCSR/SclCSC
            ptr_tuple = mat.get_c_pointers()
            ptrs = ptr_tuple[:3] + (ptr_tuple[3],)  # (data, indices, indptr, lengths)
        else:
            raise TypeError(f"Cannot extract pointers from {type(mat)}")
        
        self._cached_pointers = ptrs
        return ptrs
    
    def get_row_lengths(self) -> Array:
        """Get or compute row lengths for this chunk."""
        if self._row_lengths is None:
            mat = self.matrix
            
            # Try to get precomputed lengths
            if hasattr(mat, 'row_lengths'):
                self._row_lengths = mat.row_lengths
            else:
                # Compute from indptr (for Scipy matrices)
                if hasattr(mat, '_scl_view_indptr'):
                    indptr = mat._scl_view_indptr
                else:
                    # Wrap scipy indptr
                    indptr = Array.from_buffer(mat.indptr, 'int64', len(mat.indptr))
                
                rows = mat.shape[0]
                lengths = zeros(rows, dtype='int64')
                
                kernel_utils.compute_lengths(
                    indptr.get_pointer(),
                    rows,
                    lengths.get_pointer()
                )
                
                self._row_lengths = lengths
        
        return self._row_lengths


# =============================================================================
# Loader Decorator
# =============================================================================

def batch_loader(target_format: str):
    """
    Universal initializer for Virtual matrices.
    
    Accepts:
    - Single matrix (Scipy, SclMatrix, Virtual)
    - List of matrices (for vstack)
    
    Normalizes everything into self._chunks.
    """
    def decorator(init_func):
        @wraps(init_func)
        def wrapper(self, source: Union[Any, List[Any]], *args, **kwargs):
            self._chunks: List[_MatrixChunk] = []
            
            # Normalize to list
            sources = source if isinstance(source, (list, tuple)) else [source]
            
            self.shape = [0, 0]  # [rows, cols]
            self.dtype = None
            
            for item in sources:
                # Case 1: Recursive flattening (Virtual -> Virtual)
                if isinstance(item, (VirtualCSR, VirtualCSC)):
                    # Extract all chunks from nested virtual
                    self._chunks.extend(item._chunks)
                    
                    # Update metadata
                    if self.dtype is None:
                        self.dtype = item.dtype
                    elif self.dtype != item.dtype:
                        raise TypeError(f"Mixed dtypes: {self.dtype} vs {item.dtype}")
                    
                    # Update dimensions based on format
                    if target_format == 'CSR':
                        # vstack: same columns, accumulate rows
                        if self.shape[1] == 0:
                            self.shape[1] = item.shape[1]
                        elif self.shape[1] != item.shape[1]:
                            raise ValueError(f"Column mismatch: {self.shape[1]} vs {item.shape[1]}")
                        self.shape[0] += item.shape[0]
                    else:  # CSC
                        # hstack: same rows, accumulate columns
                        if self.shape[0] == 0:
                            self.shape[0] = item.shape[0]
                        elif self.shape[0] != item.shape[0]:
                            raise ValueError(f"Row mismatch: {self.shape[0]} vs {item.shape[0]}")
                        self.shape[1] += item.shape[1]
                    
                    continue
                
                # Case 2: Scipy matrix
                mat = item
                is_scipy = hasattr(mat, 'format')
                
                if is_scipy:
                    import scipy.sparse as sp
                    import numpy as np
                    
                    # Validate format
                    expected_fmt = target_format.lower()
                    if mat.format != expected_fmt:
                        raise TypeError(f"Expected {expected_fmt}, got {mat.format}")
                    
                    # Canonicalize
                    if not mat.has_sorted_indices:
                        mat.sort_indices()
                    mat.eliminate_zeros()
                    
                    # Cache Array views on the scipy object
                    if not hasattr(mat, '_scl_view'):
                        # Ensure int64 for indices (scipy may use int32)
                        if mat.indices.dtype != np.int64:
                            mat.indices = mat.indices.astype(np.int64)
                        if mat.indptr.dtype != np.int64:
                            mat.indptr = mat.indptr.astype(np.int64)
                        
                        mat._scl_view_data = Array.from_buffer(
                            mat.data, 
                            'float32' if mat.data.dtype == np.float32 else 'float64',
                            len(mat.data)
                        )
                        mat._scl_view_indices = Array.from_buffer(mat.indices, 'int64', len(mat.indices))
                        mat._scl_view_indptr = Array.from_buffer(mat.indptr, 'int64', len(mat.indptr))
                        mat._scl_view = True
                    
                    current_dtype = 'float32' if mat.data.dtype == np.float32 else 'float64'
                
                # Case 3: SclCSR/SclCSC
                elif hasattr(mat, 'get_c_pointers'):
                    # Validate type
                    if target_format not in type(mat).__name__:
                        raise TypeError(f"Cannot wrap {type(mat).__name__} as Virtual{target_format}")
                    
                    current_dtype = mat.dtype
                
                else:
                    raise TypeError(f"Unsupported source type: {type(mat)}")
                
                # Validate dtype consistency
                if self.dtype is None:
                    self.dtype = current_dtype
                elif self.dtype != current_dtype:
                    raise TypeError(f"Mixed dtypes: {self.dtype} vs {current_dtype}")
                
                # Validate dimensions based on format
                # CSR: vstack (vertical) - check columns match, accumulate rows
                # CSC: hstack (horizontal) - check rows match, accumulate columns
                if target_format == 'CSR':
                    # vstack: same columns, different rows
                    if self.shape[1] == 0:
                        self.shape[1] = mat.shape[1]
                    elif self.shape[1] != mat.shape[1]:
                        raise ValueError(f"Column mismatch: {self.shape[1]} vs {mat.shape[1]}")
                    self.shape[0] += mat.shape[0]
                else:  # CSC
                    # hstack: same rows, different columns
                    if self.shape[0] == 0:
                        self.shape[0] = mat.shape[0]
                    elif self.shape[0] != mat.shape[0]:
                        raise ValueError(f"Row mismatch: {self.shape[0]} vs {mat.shape[0]}")
                    self.shape[1] += mat.shape[1]
                self._chunks.append(_MatrixChunk(mat, row_map=None))
            
            self.shape = tuple(self.shape)
            
            # Call wrapped init
            return init_func(self, source, *args, **kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# VirtualCSR - Row-Oriented Sparse Matrix View
# =============================================================================

class VirtualCSR:
    """
    Zero-copy composite view for CSR matrices (row-oriented).
    
    Scipy-compatible interface with advanced features:
    - Logical vstack: Combine multiple matrices without copying
    - Zero-copy slicing: Row subsetting via indirection
    - Reference safety: Strong refs prevent GC
    - Lazy evaluation: Defer merging until to_owned()
    - Pythonic indexing: mat[rows, cols] syntax
    
    Attributes:
        shape (Tuple[int, int]): Matrix dimensions
        dtype (str): Data type ('float32' or 'float64')
        nnz (int): Number of non-zero elements
        is_contiguous (bool): True if single-block with identity map
    
    Example:
        >>> # Create from scipy
        >>> v = VirtualCSR(scipy_mat)
        >>> print(v.shape, v.nnz)
        >>> 
        >>> # Pythonic indexing
        >>> row = v[0, :]        # Get first row (dense)
        >>> subset = v[:100, :]  # First 100 rows (zero-copy view)
        >>> sub2d = v[[0,10,20], :500]  # Fancy indexing
        >>> 
        >>> # Logical stack
        >>> stacked = VirtualCSR([mat1, mat2, mat3])
        >>> 
        >>> # Chain operations
        >>> result = v.slice_rows([0, 10, 20]).slice_cols([0, 1, 2])
    """
    
    @batch_loader('CSR')
    def __init__(self, source: Union[Any, List[Any]], row_indices: Optional[Union[List[int], Array]] = None):
        """
        Initialize VirtualCSR.
        
        Args:
            source: Matrix or list of matrices
            row_indices: Optional row indices for immediate slicing
        """
        # Apply slicing if requested
        if row_indices is not None:
            self._apply_global_slice(row_indices)
    
    def _apply_global_slice(self, row_indices: Union[List[int], Array]):
        """
        Apply row slicing across all chunks.
        
        This distributes global row indices to respective chunks.
        """
        if isinstance(row_indices, list):
            row_indices = from_list(row_indices, dtype='int64')
        
        # Build chunk offsets for binary search
        offsets = [0]
        for chunk in self._chunks:
            offsets.append(offsets[-1] + chunk.shape[0])
        
        # Group indices by chunk
        chunk_indices = [[] for _ in self._chunks]
        
        for idx in row_indices.tolist():
            if idx < 0 or idx >= self.shape[0]:
                raise IndexError(f"Row index {idx} out of bounds [0, {self.shape[0]})")
            
            # Find which chunk this index belongs to
            chunk_idx = 0
            for i in range(len(offsets) - 1):
                if offsets[i] <= idx < offsets[i + 1]:
                    chunk_idx = i
                    break
            
            local_idx = idx - offsets[chunk_idx]
            chunk_indices[chunk_idx].append(local_idx)
        
        # Create new chunks with updated maps
        new_chunks = []
        for i, chunk in enumerate(self._chunks):
            if len(chunk_indices[i]) == 0:
                continue  # Skip empty chunks
            
            local_map = from_list(chunk_indices[i], dtype='int64')
            
            # Compose with existing map if present
            if chunk.row_map is not None:
                # Indirect through existing map
                composed_map = empty(len(local_map), dtype='int64')
                for j, local_idx in enumerate(local_map.tolist()):
                    composed_map[j] = chunk.row_map[local_idx]
                new_chunks.append(_MatrixChunk(chunk.matrix, composed_map))
            else:
                new_chunks.append(_MatrixChunk(chunk.matrix, local_map))
        
        self._chunks = new_chunks
        self.shape = (len(row_indices), self.shape[1])
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def rows(self) -> int:
        """Number of rows."""
        return self.shape[0]
    
    @property
    def cols(self) -> int:
        """Number of columns."""
        return self.shape[1]
    
    @property
    def nnz(self) -> int:
        """
        Approximate or exact NNZ.
        
        For multi-chunk or mapped views, this requires inspection.
        For performance, we cache or compute lazily.
        """
        if self.is_contiguous:
            return self._chunks[0].matrix.nnz
        
        # Compute by summing chunk nnz
        total = 0
        for chunk in self._chunks:
            if chunk.row_map is None:
                # Full chunk
                total += chunk.matrix.nnz
            else:
                # Need to inspect
                data_ptr, indices_ptr, indptr_ptr, _ = chunk.get_pointers()
                chunk_nnz = kernel_utils.inspect_slice_rows(
                    indptr_ptr,
                    chunk.row_map.get_pointer(),
                    len(chunk.row_map)
                )
                total += chunk_nnz
        
        return total
    
    @property
    def is_contiguous(self) -> bool:
        """True if single chunk with identity map (no indirection)."""
        return len(self._chunks) == 1 and self._chunks[0].row_map is None
    
    @property
    def is_single_block(self) -> bool:
        """True if single chunk (may have row_map)."""
        return len(self._chunks) == 1
    
    # =========================================================================
    # Access Methods (Read-Only)
    # =========================================================================
    
    def get_row(self, i: int) -> Tuple[Array, Array]:
        """
        Get row i as sparse representation.
        
        Note: For multi-chunk views, this may require chunk lookup.
        
        Args:
            i: Global row index
            
        Returns:
            (indices, values) - Column indices and values
        """
        if i < 0 or i >= self.rows:
            raise IndexError(f"Row index {i} out of bounds [0, {self.rows})")
        
        # Find which chunk contains this row
        current_offset = 0
        for chunk in self._chunks:
            if i < current_offset + chunk.shape[0]:
                # This chunk contains row i
                local_i = i - current_offset
                
                # Map through row_map if present
                if chunk.row_map is not None:
                    physical_i = chunk.row_map[local_i]
                else:
                    physical_i = local_i
                
                # Extract from backing matrix
                mat = chunk.matrix
                start = mat.indptr[physical_i]
                end = mat.indptr[physical_i + 1]
                length = end - start
                
                if length == 0:
                    return empty(0, dtype='int64'), empty(0, dtype=self.dtype)
                
                # Create copies
                row_indices = empty(length, dtype='int64')
                row_values = empty(length, dtype=self.dtype)
                
                # Extract data
                if hasattr(mat, '_scl_view_indices'):
                    # Scipy matrix
                    for k in range(length):
                        row_indices[k] = mat.indices[start + k]
                        row_values[k] = mat.data[start + k]
                else:
                    # SclMatrix
                    for k in range(length):
                        row_indices[k] = mat.indices[start + k]
                        row_values[k] = mat.data[start + k]
                
                return row_indices, row_values
            
            current_offset += chunk.shape[0]
        
        raise IndexError(f"Row {i} not found in chunks")
    
    def get_row_dense(self, i: int) -> Array:
        """
        Get row i as dense array.
        
        Args:
            i: Row index
            
        Returns:
            Dense row vector [cols]
        """
        indices, values = self.get_row(i)
        
        row = zeros(self.cols, dtype=self.dtype)
        for k in range(len(indices)):
            row[indices[k]] = values[k]
        
        return row
    
    # =========================================================================
    # Pythonic Indexing (scipy-compatible)
    # =========================================================================
    
    def __getitem__(self, key):
        """
        Pythonic indexing with scipy-compatible syntax.
        
        Supports:
        - Single row: mat[i, :] or mat[i]
        - Row slice: mat[:100, :] or mat[10:20, :]
        - Fancy indexing: mat[[0, 10, 20], :]
        - Column slicing: mat[:, :500]
        - 2D indexing: mat[[0,1,2], [5,6,7]]
        
        Args:
            key: Index, slice, or tuple of indices/slices
            
        Returns:
            Array (for single element/row) or VirtualCSR/SclCSR (for slices)
        
        Example:
            >>> v = VirtualCSR(mat)
            >>> 
            >>> # Single row (dense)
            >>> row = v[0, :]
            >>> row = v[0]  # Shorthand
            >>> 
            >>> # Row slicing (zero-copy)
            >>> subset = v[:100, :]
            >>> subset = v[10:20]
            >>> 
            >>> # Fancy indexing
            >>> subset = v[[0, 10, 20, 30], :]
            >>> 
            >>> # Column slicing (materializes)
            >>> subset = v[:, :500]
        """
        # Parse key
        if isinstance(key, tuple):
            if len(key) == 1:
                row_key = key[0]
                col_key = slice(None)
            elif len(key) == 2:
                row_key, col_key = key
            else:
                raise IndexError("Too many indices")
        else:
            # Single index: treat as row indexing
            row_key = key
            col_key = slice(None)
        
        # Process row indexing
        if isinstance(row_key, int):
            # Single row access
            if row_key < 0:
                row_key += self.rows
            if row_key < 0 or row_key >= self.rows:
                raise IndexError(f"Row index {row_key} out of bounds")
            
            # Process column indexing for single row
            if isinstance(col_key, slice):
                if col_key == slice(None):
                    # mat[i, :] - return full row as dense array
                    return self.get_row_dense(row_key)
                else:
                    # mat[i, start:stop] - return partial row
                    start, stop, step = col_key.indices(self.cols)
                    if step != 1:
                        raise NotImplementedError("Column slice with step not supported")
                    
                    row = self.get_row_dense(row_key)
                    result = empty(stop - start, dtype=self.dtype)
                    for k in range(stop - start):
                        result[k] = row[start + k]
                    return result
            elif isinstance(col_key, (list, Array)):
                # mat[i, [cols]] - fancy column indexing
                row = self.get_row_dense(row_key)
                col_indices = col_key if isinstance(col_key, Array) else from_list(col_key, dtype='int64')
                result = empty(len(col_indices), dtype=self.dtype)
                for k in range(len(col_indices)):
                    result[k] = row[col_indices[k]]
                return result
            elif isinstance(col_key, int):
                # mat[i, j] - single element
                if col_key < 0:
                    col_key += self.cols
                if col_key < 0 or col_key >= self.cols:
                    raise IndexError(f"Column index {col_key} out of bounds")
                row = self.get_row_dense(row_key)
                return row[col_key]
            else:
                raise IndexError(f"Unsupported column index type: {type(col_key)}")
        
        elif isinstance(row_key, slice):
            # Row slice: mat[start:stop, :]
            start, stop, step = row_key.indices(self.rows)
            if step != 1:
                raise NotImplementedError("Row slice with step not supported yet")
            
            row_indices = list(range(start, stop))
            sliced = self.slice_rows(row_indices)
            
            # Process column indexing
            if isinstance(col_key, slice) and col_key == slice(None):
                # mat[rows, :] - return view
                return sliced
            else:
                # mat[rows, cols] - need column slicing too
                return sliced.slice_cols(self._parse_col_indices(col_key))
        
        elif isinstance(row_key, (list, Array)):
            # Fancy row indexing: mat[[rows], :]
            row_indices = row_key if isinstance(row_key, Array) else row_key
            sliced = self.slice_rows(row_indices)
            
            # Process column indexing
            if isinstance(col_key, slice) and col_key == slice(None):
                return sliced
            else:
                return sliced.slice_cols(self._parse_col_indices(col_key))
        
        else:
            raise IndexError(f"Unsupported row index type: {type(row_key)}")
    
    def _parse_col_indices(self, col_key) -> Union[List[int], Array]:
        """Parse column indexing key into list of indices."""
        if isinstance(col_key, int):
            if col_key < 0:
                col_key += self.cols
            return [col_key]
        elif isinstance(col_key, slice):
            start, stop, step = col_key.indices(self.cols)
            if step != 1:
                raise NotImplementedError("Column slice with step not supported")
            return list(range(start, stop))
        elif isinstance(col_key, (list, Array)):
            return col_key
        else:
            raise IndexError(f"Unsupported column index type: {type(col_key)}")
    
    def __setitem__(self, key, value):
        """
        Set values (requires materialization).
        
        Virtual matrices are read-only views. To modify, first convert to owned.
        """
        raise TypeError(
            "VirtualCSR is a read-only view. "
            "Use to_owned() first to create a mutable copy."
        )
    
    def __len__(self) -> int:
        """Return number of rows (scipy compatible)."""
        return self.rows
    
    # =========================================================================
    # Slicing Operations (Explicit Methods)
    # =========================================================================
    
    def slice_rows(self, row_indices: Union[List[int], Array]):
        """
        Extract subset of rows (zero-copy when possible).
        
        Performance:
        - Single block + identity map: O(1) - creates new view
        - Single block + existing map: O(n_indices) - composes maps
        - Multi block: O(n_indices * log(n_chunks)) - dispatch + flatten
        
        Args:
            row_indices: Indices of rows to keep
            
        Returns:
            New VirtualCSR (view if single-block, may materialize for complex cases)
        
        Example:
            >>> v = VirtualCSR(scipy_mat)
            >>> subset = v.slice_rows([0, 10, 20, 30])  # Zero-copy view
            >>> 
            >>> # Chain slicing
            >>> subset2 = subset.slice_rows([0, 2])  # Composes maps
        """
        if isinstance(row_indices, list):
            row_indices = from_list(row_indices, dtype='int64')
        
        # Create new virtual object
        new_virtual = object.__new__(VirtualCSR)
        new_virtual.dtype = self.dtype
        new_virtual._chunks = []
        
        if self.is_single_block:
            # Fast path: Single chunk - create composed view
            chunk = self._chunks[0]
            
            if chunk.row_map is None:
                # Identity map -> Direct map
                new_chunk = _MatrixChunk(chunk.matrix, row_map=row_indices)
            else:
                # Compose maps: new_map[i] = old_map[indices[i]]
                composed_map = empty(len(row_indices), dtype='int64')
                for i in range(len(row_indices)):
                    idx = row_indices[i]
                    if idx < 0 or idx >= len(chunk.row_map):
                        raise IndexError(f"Index {idx} out of bounds")
                    composed_map[i] = chunk.row_map[idx]
                
                new_chunk = _MatrixChunk(chunk.matrix, row_map=composed_map)
            
            new_virtual._chunks = [new_chunk]
            new_virtual.shape = (len(row_indices), self.shape[1])
        
        else:
            # Complex path: Multi-chunk - apply global slice
            new_virtual._chunks = self._chunks.copy()
            new_virtual.shape = self.shape
            new_virtual._apply_global_slice(row_indices)
        
        return new_virtual
    
    def slice_cols(self, col_indices: Union[List[int], Array]):
        """
        Extract subset of columns (always materializes).
        
        Column slicing breaks CSR structure, requires materialization.
        
        Args:
            col_indices: Indices of columns to keep
            
        Returns:
            New SclCSR (owned matrix)
        """
        # Must materialize first, then slice
        owned = self.to_owned()
        return owned.slice_cols(col_indices)
    
    # =========================================================================
    # Materialization
    # =========================================================================
    
    def to_owned(self):
        """
        Materialize virtual view into physical SclCSR.
        
        Process:
        1. Inspect: Calculate total nnz across all chunks
        2. Allocate: Create output arrays
        3. Materialize: Copy data from all chunks (C++ kernels)
        
        Returns:
            SclCSR matrix (owned)
        
        Example:
            >>> virtual = VirtualCSR([mat1, mat2, mat3])
            >>> owned = virtual.to_owned()  # Merges into one matrix
        """
        from ._matrix import SclCSR
        
        # Step 1: Inspect - Calculate total nnz
        total_nnz = 0
        for chunk in self._chunks:
            if chunk.row_map is None:
                # Full chunk
                total_nnz += chunk.matrix.nnz
            else:
                # Sliced chunk - query C++
                data_ptr, indices_ptr, indptr_ptr, _ = chunk.get_pointers()
                chunk_nnz = kernel_utils.inspect_slice_rows(
                    indptr_ptr,
                    chunk.row_map.get_pointer(),
                    len(chunk.row_map)
                )
                total_nnz += chunk_nnz
        
        # Step 2: Allocate
        out_data = empty(total_nnz, dtype=self.dtype)
        out_indices = empty(total_nnz, dtype='int64')
        out_indptr = zeros(self.rows + 1, dtype='int64')
        
        # Step 3: Materialize - Process each chunk
        current_row_offset = 0
        current_nnz_offset = 0
        
        for chunk in self._chunks:
            n_rows = chunk.shape[0]
            
            # Get source pointers
            src_data_ptr, src_indices_ptr, src_indptr_ptr, _ = chunk.get_pointers()
            
            # Get destination pointers (offset into output arrays)
            # Note: We can't do pointer arithmetic with ctypes.c_void_p directly
            # Need to use ctypes.cast
            
            dst_data_ptr = ctypes.cast(
                out_data.get_pointer().value + current_nnz_offset * out_data.itemsize,
                type(out_data.get_pointer())
            )
            dst_indices_ptr = ctypes.cast(
                out_indices.get_pointer().value + current_nnz_offset * out_indices.itemsize,
                type(out_indices.get_pointer())
            )
            dst_indptr_ptr = ctypes.cast(
                out_indptr.get_pointer().value + current_row_offset * out_indptr.itemsize,
                type(out_indptr.get_pointer())
            )
            
            if chunk.row_map is None:
                # Full chunk copy - need custom kernel for indptr offset
                # For now, use Python loop for indptr (small overhead)
                
                # Copy data and indices (bulk)
                chunk_nnz = chunk.matrix.nnz
                mat = chunk.matrix
                
                # Direct memory copy
                if hasattr(mat, 'data'):
                    # Scipy
                    ctypes.memmove(
                        dst_data_ptr,
                        mat.data.ctypes.data,
                        chunk_nnz * out_data.itemsize
                    )
                    ctypes.memmove(
                        dst_indices_ptr,
                        mat.indices.ctypes.data,
                        chunk_nnz * out_indices.itemsize
                    )
                else:
                    # SclMatrix
                    ctypes.memmove(
                        dst_data_ptr,
                        mat.data.get_pointer(),
                        chunk_nnz * out_data.itemsize
                    )
                    ctypes.memmove(
                        dst_indices_ptr,
                        mat.indices.get_pointer(),
                        chunk_nnz * out_indices.itemsize
                    )
                
                # Adjust indptr
                base_offset = mat.indptr[0] if hasattr(mat, 'indptr') else mat.indptr[0]
                for j in range(n_rows + 1):
                    src_val = mat.indptr[j] if hasattr(mat, 'indptr') else mat.indptr[j]
                    out_indptr[current_row_offset + j] = src_val - base_offset + current_nnz_offset
                
                current_nnz_offset += chunk_nnz
            
            else:
                # Sliced chunk - use materialize kernel
                kernel_utils.materialize_slice_rows(
                    src_data_ptr,
                    src_indices_ptr,
                    src_indptr_ptr,
                    chunk.row_map.get_pointer(),
                    n_rows,
                    dst_data_ptr,
                    dst_indices_ptr,
                    dst_indptr_ptr
                )
                
                # Read back how much we wrote
                chunk_nnz = out_indptr[current_row_offset + n_rows] - current_nnz_offset
                current_nnz_offset += chunk_nnz
            
            current_row_offset += n_rows
        
        return SclCSR(out_data, out_indices, out_indptr, shape=self.shape)
    
    # =========================================================================
    # Conversion and Format Operations
    # =========================================================================
    
    def copy(self):
        """
        Create a deep copy (materializes to owned matrix).
        
        Returns:
            SclCSR matrix (owned copy)
        """
        return self.to_owned()
    
    def tocsr(self):
        """
        Convert to CSR format (identity operation).
        
        Returns:
            self (already CSR) or SclCSR if materialization needed
        """
        if self.is_contiguous:
            return self
        return self.to_owned()
    
    def tocsc(self):
        """
        Convert to CSC format.
        
        Returns:
            SclCSC matrix
        """
        return self.to_owned().tocsc()
    
    def toarray(self):
        """
        Convert to dense array.
        
        Returns:
            Dense 2D array (requires scipy/numpy)
        """
        return self.to_scipy().toarray()
    
    def todense(self):
        """Convert to dense (alias for toarray)."""
        return self.toarray()
    
    def to_scipy(self):
        """
        Convert to scipy.sparse.csr_matrix.
        
        Returns:
            scipy CSR matrix
        """
        return self.to_owned().to_scipy()
    
    def transpose(self):
        """
        Transpose matrix (CSR → CSC).
        
        Returns:
            VirtualCSC (transposed)
        """
        return VirtualCSC(self.to_owned().tocsc())
    
    @property
    def T(self):
        """Transpose property (scipy compatible)."""
        return self.transpose()
    
    # =========================================================================
    # Matrix Properties (scipy-compatible)
    # =========================================================================
    
    def getnnz(self, axis: Optional[int] = None):
        """
        Get number of non-zeros.
        
        Args:
            axis: None (total), 0 (per column), 1 (per row)
            
        Returns:
            int (if axis=None) or Array (if axis specified)
        """
        if axis is None:
            return self.nnz
        elif axis == 0:
            # NNZ per column
            result = zeros(self.cols, dtype='int64')
            # Would need a kernel for this
            raise NotImplementedError("getnnz(axis=0) not yet implemented")
        elif axis == 1:
            # NNZ per row
            result = zeros(self.rows, dtype='int64')
            # Would need a kernel for this
            raise NotImplementedError("getnnz(axis=1) not yet implemented")
        else:
            raise ValueError(f"axis must be None, 0, or 1, got {axis}")
    
    def get_shape(self) -> Tuple[int, int]:
        """Get shape (scipy compatible)."""
        return self.shape
    
    def getformat(self) -> str:
        """Get format name (scipy compatible)."""
        return 'csr'
    
    @property
    def format(self) -> str:
        """Format name (scipy compatible)."""
        return 'csr'
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def __repr__(self) -> str:
        chunk_info = f"{len(self._chunks)} chunks" if len(self._chunks) > 1 else "1 chunk"
        view_info = "view" if not self.is_contiguous else "contiguous"
        return (f"VirtualCSR(shape={self.shape}, {chunk_info}, "
                f"dtype={self.dtype}, {view_info})")
    
    def get_c_pointers(self) -> Optional[Tuple]:
        """
        Get C pointers (only for contiguous single-block views).
        
        Returns:
            (data_ptr, indices_ptr, indptr_ptr, row_lengths_ptr, rows, cols)
            or None if not representable as single block
        
        Note: Most C kernels can only work with contiguous matrices.
              Use to_owned() first for multi-block cases.
        """
        if not self.is_contiguous:
            raise RuntimeError(
                "Cannot get C pointers from multi-chunk virtual matrix. "
                "Call to_owned() first to materialize."
            )
        
        chunk = self._chunks[0]
        ptrs = chunk.get_pointers()
        
        return (
            ptrs[0],  # data
            ptrs[1],  # indices
            ptrs[2],  # indptr
            ptrs[3] if ptrs[3] else None,  # row_lengths
            self.rows,
            self.cols
        )


# =============================================================================
# VirtualCSC - Column-Oriented Sparse Matrix View
# =============================================================================

class VirtualCSC:
    """
    Zero-copy composite view for CSC matrices (column-oriented).
    
    Scipy-compatible interface optimized for gene-wise operations:
    - Logical hstack: Combine multiple matrices without copying
    - Zero-copy slicing: Column subsetting via indirection
    - Reference safety: Strong refs prevent GC
    - Lazy evaluation: Defer merging until to_owned()
    - Pythonic indexing: mat[rows, cols] syntax
    
    Attributes:
        shape (Tuple[int, int]): Matrix dimensions
        dtype (str): Data type ('float32' or 'float64')
        nnz (int): Number of non-zero elements
        is_contiguous (bool): True if single-block with identity map
    
    Example:
        >>> # Create from scipy
        >>> v = VirtualCSC(scipy_mat)
        >>> 
        >>> # Pythonic indexing
        >>> col = v[:, 0]         # Get first column (dense)
        >>> subset = v[:, :100]   # First 100 columns (zero-copy view)
        >>> 
        >>> # Logical stack (horizontal)
        >>> stacked = VirtualCSC([mat1, mat2, mat3])
        >>> 
        >>> # Access genes
        >>> gene_expr = v[:, 42]  # Expression of gene 42 across all cells
    """
    
    @batch_loader('CSC')
    def __init__(self, source: Union[Any, List[Any]], col_indices: Optional[Union[List[int], Array]] = None):
        """
        Initialize VirtualCSC.
        
        Args:
            source: Matrix or list of matrices (for hstack)
            col_indices: Optional column indices for immediate slicing
        """
        if col_indices is not None:
            self._apply_global_slice(col_indices)
    
    def _apply_global_slice(self, col_indices: Union[List[int], Array]):
        """Apply column slicing across all chunks."""
        if isinstance(col_indices, list):
            col_indices = from_list(col_indices, dtype='int64')
        
        # Similar logic to VirtualCSR but for columns
        offsets = [0]
        for chunk in self._chunks:
            offsets.append(offsets[-1] + chunk.shape[1])
        
        chunk_indices = [[] for _ in self._chunks]
        
        for idx in col_indices.tolist():
            if idx < 0 or idx >= self.shape[1]:
                raise IndexError(f"Column index {idx} out of bounds")
            
            chunk_idx = 0
            for i in range(len(offsets) - 1):
                if offsets[i] <= idx < offsets[i + 1]:
                    chunk_idx = i
                    break
            
            local_idx = idx - offsets[chunk_idx]
            chunk_indices[chunk_idx].append(local_idx)
        
        new_chunks = []
        for i, chunk in enumerate(self._chunks):
            if len(chunk_indices[i]) == 0:
                continue
            
            local_map = from_list(chunk_indices[i], dtype='int64')
            
            if chunk.row_map is not None:  # row_map used as col_map for CSC
                composed_map = empty(len(local_map), dtype='int64')
                for j, local_idx in enumerate(local_map.tolist()):
                    composed_map[j] = chunk.row_map[local_idx]
                new_chunks.append(_MatrixChunk(chunk.matrix, composed_map))
            else:
                new_chunks.append(_MatrixChunk(chunk.matrix, local_map))
        
        self._chunks = new_chunks
        self.shape = (self.shape[0], len(col_indices))
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def rows(self) -> int:
        return self.shape[0]
    
    @property
    def cols(self) -> int:
        return self.shape[1]
    
    @property
    def nnz(self) -> int:
        """Approximate or exact NNZ."""
        if self.is_contiguous:
            return self._chunks[0].matrix.nnz
        
        total = 0
        for chunk in self._chunks:
            if chunk.row_map is None:
                total += chunk.matrix.nnz
            else:
                data_ptr, indices_ptr, indptr_ptr, _ = chunk.get_pointers()
                chunk_nnz = kernel_utils.inspect_slice_rows(
                    indptr_ptr,
                    chunk.row_map.get_pointer(),
                    len(chunk.row_map)
                )
                total += chunk_nnz
        
        return total
    
    @property
    def is_contiguous(self) -> bool:
        return len(self._chunks) == 1 and self._chunks[0].row_map is None
    
    @property
    def is_single_block(self) -> bool:
        return len(self._chunks) == 1
    
    # =========================================================================
    # Pythonic Indexing (scipy-compatible)
    # =========================================================================
    
    def __getitem__(self, key):
        """
        Pythonic indexing with scipy-compatible syntax.
        
        Supports:
        - Single column: mat[:, j] or mat[:, j:j+1]
        - Column slice: mat[:, :100] or mat[:, 10:20]
        - Fancy indexing: mat[:, [0, 10, 20]]
        - Row slicing: mat[:500, :]
        - 2D indexing: mat[[0,1,2], [5,6,7]]
        
        Args:
            key: Index, slice, or tuple of indices/slices
            
        Returns:
            Array (for single column) or VirtualCSC/SclCSC (for slices)
        
        Example:
            >>> v = VirtualCSC(mat)
            >>> 
            >>> # Single column (dense)
            >>> col = v[:, 0]
            >>> 
            >>> # Column slicing (zero-copy)
            >>> subset = v[:, :100]
            >>> 
            >>> # Fancy indexing
            >>> subset = v[:, [0, 10, 20, 30]]
            >>> 
            >>> # Row slicing (materializes for CSC)
            >>> subset = v[:500, :]
        """
        # Parse key
        if isinstance(key, tuple):
            if len(key) == 1:
                row_key = key[0]
                col_key = slice(None)
            elif len(key) == 2:
                row_key, col_key = key
            else:
                raise IndexError("Too many indices")
        else:
            # Single index: ambiguous for CSC, treat as column for consistency
            row_key = slice(None)
            col_key = key
        
        # Process column indexing (primary for CSC)
        if isinstance(col_key, int):
            # Single column access
            if col_key < 0:
                col_key += self.cols
            if col_key < 0 or col_key >= self.cols:
                raise IndexError(f"Column index {col_key} out of bounds")
            
            # Process row indexing for single column
            if isinstance(row_key, slice) and row_key == slice(None):
                # mat[:, j] - return full column as dense array
                return self.get_col_dense(col_key)
            elif isinstance(row_key, slice):
                # mat[start:stop, j] - return partial column
                start, stop, step = row_key.indices(self.rows)
                if step != 1:
                    raise NotImplementedError("Row slice with step not supported")
                
                col = self.get_col_dense(col_key)
                result = empty(stop - start, dtype=self.dtype)
                for k in range(stop - start):
                    result[k] = col[start + k]
                return result
            elif isinstance(row_key, (list, Array)):
                # mat[[rows], j] - fancy row indexing on column
                col = self.get_col_dense(col_key)
                row_indices = row_key if isinstance(row_key, Array) else from_list(row_key, dtype='int64')
                result = empty(len(row_indices), dtype=self.dtype)
                for k in range(len(row_indices)):
                    result[k] = col[row_indices[k]]
                return result
            elif isinstance(row_key, int):
                # mat[i, j] - single element
                if row_key < 0:
                    row_key += self.rows
                if row_key < 0 or row_key >= self.rows:
                    raise IndexError(f"Row index {row_key} out of bounds")
                col = self.get_col_dense(col_key)
                return col[row_key]
            else:
                raise IndexError(f"Unsupported row index type: {type(row_key)}")
        
        elif isinstance(col_key, slice):
            # Column slice: mat[:, start:stop]
            start, stop, step = col_key.indices(self.cols)
            if step != 1:
                raise NotImplementedError("Column slice with step not supported yet")
            
            col_indices = list(range(start, stop))
            sliced = self.slice_cols(col_indices)
            
            # Process row indexing
            if isinstance(row_key, slice) and row_key == slice(None):
                # mat[:, cols] - return view
                return sliced
            else:
                # mat[rows, cols] - need row slicing too (materializes)
                return sliced.slice_rows(self._parse_row_indices(row_key))
        
        elif isinstance(col_key, (list, Array)):
            # Fancy column indexing: mat[:, [cols]]
            col_indices = col_key if isinstance(col_key, Array) else col_key
            sliced = self.slice_cols(col_indices)
            
            # Process row indexing
            if isinstance(row_key, slice) and row_key == slice(None):
                return sliced
            else:
                return sliced.slice_rows(self._parse_row_indices(row_key))
        
        else:
            raise IndexError(f"Unsupported column index type: {type(col_key)}")
    
    def _parse_row_indices(self, row_key) -> Union[List[int], Array]:
        """Parse row indexing key into list of indices."""
        if isinstance(row_key, int):
            if row_key < 0:
                row_key += self.rows
            return [row_key]
        elif isinstance(row_key, slice):
            start, stop, step = row_key.indices(self.rows)
            if step != 1:
                raise NotImplementedError("Row slice with step not supported")
            return list(range(start, stop))
        elif isinstance(row_key, (list, Array)):
            return row_key
        else:
            raise IndexError(f"Unsupported row index type: {type(row_key)}")
    
    def __setitem__(self, key, value):
        """
        Set values (requires materialization).
        
        Virtual matrices are read-only views. To modify, first convert to owned.
        """
        raise TypeError(
            "VirtualCSC is a read-only view. "
            "Use to_owned() first to create a mutable copy."
        )
    
    def __len__(self) -> int:
        """Return number of rows (scipy compatible)."""
        return self.rows
    
    # =========================================================================
    # Access Methods (Explicit)
    # =========================================================================
    
    def get_col(self, j: int) -> Tuple[Array, Array]:
        """
        Get column j as sparse representation.
        
        Args:
            j: Global column index
            
        Returns:
            (indices, values) - Row indices and values
        """
        if j < 0 or j >= self.cols:
            raise IndexError(f"Column index {j} out of bounds [0, {self.cols})")
        
        current_offset = 0
        for chunk in self._chunks:
            if j < current_offset + chunk.shape[1]:
                local_j = j - current_offset
                
                if chunk.row_map is not None:  # col_map for CSC
                    physical_j = chunk.row_map[local_j]
                else:
                    physical_j = local_j
                
                mat = chunk.matrix
                start = mat.indptr[physical_j]
                end = mat.indptr[physical_j + 1]
                length = end - start
                
                if length == 0:
                    return empty(0, dtype='int64'), empty(0, dtype=self.dtype)
                
                col_indices = empty(length, dtype='int64')
                col_values = empty(length, dtype=self.dtype)
                
                if hasattr(mat, '_scl_view_indices'):
                    for k in range(length):
                        col_indices[k] = mat.indices[start + k]
                        col_values[k] = mat.data[start + k]
                else:
                    for k in range(length):
                        col_indices[k] = mat.indices[start + k]
                        col_values[k] = mat.data[start + k]
                
                return col_indices, col_values
            
            current_offset += chunk.shape[1]
        
        raise IndexError(f"Column {j} not found in chunks")
    
    def get_col_dense(self, j: int) -> Array:
        """Get column j as dense array."""
        indices, values = self.get_col(j)
        
        col = zeros(self.rows, dtype=self.dtype)
        for k in range(len(indices)):
            col[indices[k]] = values[k]
        
        return col
    
    # =========================================================================
    # Slicing Operations
    # =========================================================================
    
    def slice_cols(self, col_indices: Union[List[int], Array]):
        """
        Extract subset of columns (zero-copy when possible).
        
        Args:
            col_indices: Indices of columns to keep
            
        Returns:
            New VirtualCSC (view if single-block)
        """
        if isinstance(col_indices, list):
            col_indices = from_list(col_indices, dtype='int64')
        
        new_virtual = object.__new__(VirtualCSC)
        new_virtual.dtype = self.dtype
        new_virtual._chunks = []
        
        if self.is_single_block:
            chunk = self._chunks[0]
            
            if chunk.row_map is None:
                new_chunk = _MatrixChunk(chunk.matrix, row_map=col_indices)
            else:
                composed_map = empty(len(col_indices), dtype='int64')
                for i in range(len(col_indices)):
                    idx = col_indices[i]
                    if idx < 0 or idx >= len(chunk.row_map):
                        raise IndexError(f"Index {idx} out of bounds")
                    composed_map[i] = chunk.row_map[idx]
                
                new_chunk = _MatrixChunk(chunk.matrix, row_map=composed_map)
            
            new_virtual._chunks = [new_chunk]
            new_virtual.shape = (self.shape[0], len(col_indices))
        
        else:
            new_virtual._chunks = self._chunks.copy()
            new_virtual.shape = self.shape
            new_virtual._apply_global_slice(col_indices)
        
        return new_virtual
    
    def slice_rows(self, row_indices: Union[List[int], Array]):
        """
        Extract subset of rows (always materializes for CSC).
        
        Row slicing breaks CSC structure.
        """
        owned = self.to_owned()
        # CSC doesn't have slice_rows in SclCSC yet, need to implement or convert
        # For now, convert to CSR, slice, convert back
        return owned.tocsr().slice_rows(row_indices).tocsc()
    
    # =========================================================================
    # Materialization
    # =========================================================================
    
    def to_owned(self):
        """
        Materialize virtual view into physical SclCSC.
        
        Returns:
            SclCSC matrix (owned)
        """
        from ._matrix import SclCSC
        
        # Similar logic to VirtualCSR.to_owned()
        # But for CSC, we iterate over columns instead of rows
        
        total_nnz = 0
        for chunk in self._chunks:
            if chunk.row_map is None:
                total_nnz += chunk.matrix.nnz
            else:
                data_ptr, indices_ptr, indptr_ptr, _ = chunk.get_pointers()
                chunk_nnz = kernel_utils.inspect_slice_rows(
                    indptr_ptr,
                    chunk.row_map.get_pointer(),
                    len(chunk.row_map)
                )
                total_nnz += chunk_nnz
        
        out_data = empty(total_nnz, dtype=self.dtype)
        out_indices = empty(total_nnz, dtype='int64')
        out_indptr = zeros(self.cols + 1, dtype='int64')
        
        current_col_offset = 0
        current_nnz_offset = 0
        
        for chunk in self._chunks:
            n_cols = chunk.shape[1]
            src_data_ptr, src_indices_ptr, src_indptr_ptr, _ = chunk.get_pointers()
            
            dst_data_ptr = ctypes.cast(
                out_data.get_pointer().value + current_nnz_offset * out_data.itemsize,
                type(out_data.get_pointer())
            )
            dst_indices_ptr = ctypes.cast(
                out_indices.get_pointer().value + current_nnz_offset * out_indices.itemsize,
                type(out_indices.get_pointer())
            )
            dst_indptr_ptr = ctypes.cast(
                out_indptr.get_pointer().value + current_col_offset * out_indptr.itemsize,
                type(out_indptr.get_pointer())
            )
            
            if chunk.row_map is None:
                chunk_nnz = chunk.matrix.nnz
                mat = chunk.matrix
                
                if hasattr(mat, 'data'):
                    ctypes.memmove(dst_data_ptr, mat.data.ctypes.data, chunk_nnz * out_data.itemsize)
                    ctypes.memmove(dst_indices_ptr, mat.indices.ctypes.data, chunk_nnz * out_indices.itemsize)
                else:
                    ctypes.memmove(dst_data_ptr, mat.data.get_pointer(), chunk_nnz * out_data.itemsize)
                    ctypes.memmove(dst_indices_ptr, mat.indices.get_pointer(), chunk_nnz * out_indices.itemsize)
                
                base_offset = mat.indptr[0]
                for j in range(n_cols + 1):
                    out_indptr[current_col_offset + j] = mat.indptr[j] - base_offset + current_nnz_offset
                
                current_nnz_offset += chunk_nnz
            
            else:
                kernel_utils.materialize_slice_rows(
                    src_data_ptr, src_indices_ptr, src_indptr_ptr,
                    chunk.row_map.get_pointer(), n_cols,
                    dst_data_ptr, dst_indices_ptr, dst_indptr_ptr
                )
                
                chunk_nnz = out_indptr[current_col_offset + n_cols] - current_nnz_offset
                current_nnz_offset += chunk_nnz
            
            current_col_offset += n_cols
        
        return SclCSC(out_data, out_indices, out_indptr, shape=self.shape)
    
    # =========================================================================
    # Conversion and Format Operations
    # =========================================================================
    
    def copy(self):
        """
        Create a deep copy (materializes to owned matrix).
        
        Returns:
            SclCSC matrix (owned copy)
        """
        return self.to_owned()
    
    def tocsc(self):
        """
        Convert to CSC format (identity operation).
        
        Returns:
            self (already CSC) or SclCSC if materialization needed
        """
        if self.is_contiguous:
            return self
        return self.to_owned()
    
    def tocsr(self):
        """
        Convert to CSR format.
        
        Returns:
            SclCSR matrix
        """
        return self.to_owned().tocsr()
    
    def toarray(self):
        """
        Convert to dense array.
        
        Returns:
            Dense 2D array (requires scipy/numpy)
        """
        return self.to_scipy().toarray()
    
    def todense(self):
        """Convert to dense (alias for toarray)."""
        return self.toarray()
    
    def to_scipy(self):
        """
        Convert to scipy.sparse.csc_matrix.
        
        Returns:
            scipy CSC matrix
        """
        return self.to_owned().to_scipy()
    
    def transpose(self):
        """
        Transpose matrix (CSC → CSR).
        
        Returns:
            VirtualCSR (transposed)
        """
        return VirtualCSR(self.to_owned().tocsr())
    
    @property
    def T(self):
        """Transpose property (scipy compatible)."""
        return self.transpose()
    
    # =========================================================================
    # Matrix Properties (scipy-compatible)
    # =========================================================================
    
    def getnnz(self, axis: Optional[int] = None):
        """
        Get number of non-zeros.
        
        Args:
            axis: None (total), 0 (per column), 1 (per row)
            
        Returns:
            int (if axis=None) or Array (if axis specified)
        """
        if axis is None:
            return self.nnz
        elif axis == 0:
            # NNZ per column (natural for CSC)
            if self.is_contiguous:
                chunk = self._chunks[0]
                lengths = chunk.get_row_lengths()  # For CSC, this is col_lengths
                return lengths
            else:
                # Need to materialize or compute
                raise NotImplementedError("getnnz(axis=0) for multi-chunk not yet implemented")
        elif axis == 1:
            # NNZ per row
            raise NotImplementedError("getnnz(axis=1) not yet implemented")
        else:
            raise ValueError(f"axis must be None, 0, or 1, got {axis}")
    
    def get_shape(self) -> Tuple[int, int]:
        """Get shape (scipy compatible)."""
        return self.shape
    
    def getformat(self) -> str:
        """Get format name (scipy compatible)."""
        return 'csc'
    
    @property
    def format(self) -> str:
        """Format name (scipy compatible)."""
        return 'csc'
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def __repr__(self) -> str:
        chunk_info = f"{len(self._chunks)} chunks" if len(self._chunks) > 1 else "1 chunk"
        view_info = "view" if not self.is_contiguous else "contiguous"
        return (f"VirtualCSC(shape={self.shape}, {chunk_info}, "
                f"dtype={self.dtype}, {view_info})")
    
    def get_c_pointers(self) -> Optional[Tuple]:
        """Get C pointers (only for contiguous single-block views)."""
        if not self.is_contiguous:
            raise RuntimeError(
                "Cannot get C pointers from multi-chunk virtual matrix. "
                "Call to_owned() first to materialize."
            )
        
        chunk = self._chunks[0]
        ptrs = chunk.get_pointers()
        
        return (
            ptrs[0], ptrs[1], ptrs[2],
            ptrs[3] if ptrs[3] else None,
            self.rows, self.cols
        )
