"""
Custom Sparse Matrix Classes (SclCSR/SclCSC)

High-performance implementation with C++ kernel delegation.
Python handles business logic, C++ handles computational hot paths.
"""

from typing import Optional, Tuple, Any, List, Union
from ._array import Array, zeros, from_list, empty
from .._kernel import utils as kernel_utils

__all__ = ['SclCSR', 'SclCSC']


class SclCSR:
    """
    SCL Custom CSR Matrix (Cells × Genes).
    
    Format:
    - data: Non-zero values [nnz]
    - indices: Column indices [nnz]
    - indptr: Row pointers [rows + 1]
    - row_lengths: Number of non-zeros per row [rows]
    
    The row_lengths array enables O(1) row length queries and optimizes
    many kernel operations.
    
    Attributes:
        data (Array): Non-zero values
        indices (Array): Column indices
        indptr (Array): Row pointers
        row_lengths (Array): Row lengths
        shape (Tuple[int, int]): Matrix dimensions (rows, cols)
    """
    
    def __init__(
        self,
        data: Array,
        indices: Array,
        indptr: Array,
        shape: Tuple[int, int],
        row_lengths: Optional[Array] = None
    ):
        """
        Initialize SclCSR matrix.
        
        Args:
            data: Non-zero values (dtype: float32 or float64)
            indices: Column indices (dtype: int64)
            indptr: Row pointers (dtype: int64)
            shape: Matrix dimensions (rows, cols)
            row_lengths: Optional precomputed row lengths (dtype: int64)
        """
        # Type validation
        if data.dtype not in ('float32', 'float64'):
            raise TypeError(f"data must be float32/float64, got {data.dtype}")
        if indices.dtype != 'int64':
            raise TypeError(f"indices must be int64, got {indices.dtype}")
        if indptr.dtype != 'int64':
            raise TypeError(f"indptr must be int64, got {indptr.dtype}")
        
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = tuple(shape)
        
        # Validate dimensions
        rows, cols = self.shape
        if rows < 0 or cols < 0:
            raise ValueError(f"Invalid shape: {self.shape}")
        
        if len(self.indptr) != rows + 1:
            raise ValueError(f"indptr size mismatch: expected {rows+1}, got {len(self.indptr)}")
        
        nnz = len(self.data)
        if len(self.indices) != nnz:
            raise ValueError(f"indices size mismatch: expected {nnz}, got {len(self.indices)}")
        
        # Validate that indptr[-1] matches data size (unless it's an empty matrix)
        # For empty() allocated matrices, indptr may be all zeros
        if len(self.indptr) > 0:
            last_indptr = self.indptr[-1]
            # Only validate if indptr has been initialized (not all zeros or equals nnz)
            if last_indptr != 0 and last_indptr != nnz:
                raise ValueError(
                    f"Data size mismatch: indptr[-1]={last_indptr} but nnz={nnz}"
                )
        
        # Compute or validate row_lengths
        if row_lengths is None:
            self.row_lengths = self._compute_row_lengths()
        else:
            if row_lengths.dtype != 'int64':
                raise TypeError(f"row_lengths must be int64, got {row_lengths.dtype}")
            if len(row_lengths) != rows:
                raise ValueError(f"row_lengths size mismatch: expected {rows}, got {len(row_lengths)}")
            self.row_lengths = row_lengths
    
    def _compute_row_lengths(self) -> Array:
        """
        Compute row lengths from indptr (delegated to C++).
        
        This uses parallel diff operation in C++, ~100x faster than Python loop.
        """
        rows = self.shape[0]
        
        # Handle empty matrix edge case
        if rows == 0:
            return zeros(0, dtype='int64')
        
        lengths = zeros(rows, dtype='int64')
        
        # Call C++ kernel: parallel diff operation
        kernel_utils.compute_lengths(
            self.indptr.get_pointer(),
            rows,
            lengths.get_pointer()
        )
        
        return lengths
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.data)
    
    @property
    def rows(self) -> int:
        """Number of rows."""
        return self.shape[0]
    
    @property
    def cols(self) -> int:
        """Number of columns."""
        return self.shape[1]
    
    @property
    def dtype(self) -> str:
        """Data type."""
        return self.data.dtype
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def empty(cls, rows: int, cols: int, nnz: int, dtype: str = 'float64') -> 'SclCSR':
        """
        Create empty CSR matrix with pre-allocated arrays.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            nnz: Number of non-zeros to allocate
            dtype: Data type ('float32' or 'float64')
            
        Returns:
            Empty SclCSR matrix
        
        Example:
            >>> mat = SclCSR.empty(1000, 2000, 50000, dtype='float32')
            >>> # Fill in data, indices, indptr manually
        """
        data = empty(nnz, dtype=dtype)
        indices = empty(nnz, dtype='int64')
        indptr = zeros(rows + 1, dtype='int64')
        
        return cls(data, indices, indptr, shape=(rows, cols))
    
    @classmethod
    def zeros(cls, rows: int, cols: int, dtype: str = 'float64') -> 'SclCSR':
        """
        Create zero matrix (no non-zero elements).
        
        Args:
            rows: Number of rows
            cols: Number of columns
            dtype: Data type
            
        Returns:
            Zero SclCSR matrix
        
        Example:
            >>> mat = SclCSR.zeros(1000, 2000)
            >>> print(mat.nnz)  # 0
        """
        data = empty(0, dtype=dtype)
        indices = empty(0, dtype='int64')
        indptr = zeros(rows + 1, dtype='int64')
        
        return cls(data, indices, indptr, shape=(rows, cols))
    
    @classmethod
    def from_dense(cls, dense: List[List[float]], dtype: str = 'float64') -> 'SclCSR':
        """
        Create CSR matrix from dense 2D list.
        
        Args:
            dense: 2D list (rows × cols)
            dtype: Data type
            
        Returns:
            SclCSR matrix
        
        Example:
            >>> dense = [[1.0, 0.0, 2.0],
            ...          [0.0, 3.0, 0.0]]
            >>> mat = SclCSR.from_dense(dense)
        """
        rows = len(dense)
        if rows == 0:
            return cls.zeros(0, 0, dtype)
        
        cols = len(dense[0])
        
        # Count non-zeros
        nnz = sum(1 for row in dense for val in row if val != 0.0)
        
        # Allocate arrays
        data_list = []
        indices_list = []
        indptr_list = [0]
        
        for row in dense:
            for j, val in enumerate(row):
                if val != 0.0:
                    data_list.append(val)
                    indices_list.append(j)
            indptr_list.append(len(data_list))
        
        data = from_list(data_list, dtype=dtype)
        indices = from_list(indices_list, dtype='int64')
        indptr = from_list(indptr_list, dtype='int64')
        
        return cls(data, indices, indptr, shape=(rows, cols))
    
    @classmethod
    def from_scipy(cls, mat: Any) -> 'SclCSR':
        """
        Create SclCSR from scipy.sparse.csr_matrix.
        
        Args:
            mat: scipy CSR matrix
            
        Returns:
            SclCSR matrix
        """
        try:
            import scipy.sparse as sp
        except ImportError:
            raise ImportError("scipy is required for from_scipy()")
        
        if not sp.isspmatrix_csr(mat):
            raise TypeError("Input must be scipy.sparse.csr_matrix")
        
        # Ensure canonical format
        mat.sort_indices()
        mat.eliminate_zeros()
        
        # Convert to our Array format
        import numpy as np
        
        # Determine dtype
        if mat.data.dtype == np.float32:
            dtype = 'float32'
        else:
            dtype = 'float64'
        
        # Ensure int64 for indices (required by C API)
        if mat.indices.dtype != np.int64:
            mat.indices = mat.indices.astype(np.int64)
        if mat.indptr.dtype != np.int64:
            mat.indptr = mat.indptr.astype(np.int64)
        
        # Create Arrays from numpy arrays
        data = Array.from_buffer(mat.data, dtype, len(mat.data))
        indices = Array.from_buffer(mat.indices, 'int64', len(mat.indices))
        indptr = Array.from_buffer(mat.indptr, 'int64', len(mat.indptr))
        
        return cls(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=mat.shape
        )
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_scipy(self) -> Any:
        """
        Convert to scipy.sparse.csr_matrix.
        
        Returns:
            scipy CSR matrix
        """
        try:
            import scipy.sparse as sp
            import numpy as np
        except ImportError:
            raise ImportError("scipy and numpy are required for to_scipy()")
        
        # Convert Arrays to numpy arrays
        np_dtype = np.float32 if self.data.dtype == 'float32' else np.float64
        
        data_np = np.frombuffer(self.data.tobytes(), dtype=np_dtype)
        indices_np = np.frombuffer(self.indices.tobytes(), dtype=np.int64)
        indptr_np = np.frombuffer(self.indptr.tobytes(), dtype=np.int64)
        
        return sp.csr_matrix(
            (data_np, indices_np, indptr_np),
            shape=self.shape
        )
    
    def tocsc(self) -> 'SclCSC':
        """
        Convert to CSC format.
        
        Returns:
            SclCSC matrix
        
        Note: This uses scipy as intermediate, requires scipy/numpy.
        """
        scipy_csr = self.to_scipy()
        scipy_csc = scipy_csr.tocsc()
        return SclCSC.from_scipy(scipy_csc)
    
    def copy(self) -> 'SclCSR':
        """Create a deep copy."""
        return SclCSR(
            data=self.data.copy(),
            indices=self.indices.copy(),
            indptr=self.indptr.copy(),
            shape=self.shape,
            row_lengths=self.row_lengths.copy()
        )
    
    # =========================================================================
    # Slicing Operations
    # =========================================================================
    
    def get_row(self, i: int) -> Tuple[Array, Array]:
        """
        Get row i as sparse representation (indices, values).
        
        Args:
            i: Row index
            
        Returns:
            (indices, values) - Column indices and values for non-zeros
        
        Example:
            >>> indices, values = mat.get_row(0)
            >>> print(f"Row 0 has {len(values)} non-zeros")
        """
        if i < 0 or i >= self.rows:
            raise IndexError(f"Row index {i} out of bounds [0, {self.rows})")
        
        start = self.indptr[i]
        end = self.indptr[i + 1]
        length = end - start
        
        if length == 0:
            return empty(0, dtype='int64'), empty(0, dtype=self.dtype)
        
        # Create views (zero-copy slicing)
        row_indices = empty(length, dtype='int64')
        row_values = empty(length, dtype=self.dtype)
        
        for k in range(length):
            row_indices[k] = self.indices[start + k]
            row_values[k] = self.data[start + k]
        
        return row_indices, row_values
    
    def get_row_dense(self, i: int) -> Array:
        """
        Get row i as dense array.
        
        Args:
            i: Row index
            
        Returns:
            Dense row vector [cols]
        
        Example:
            >>> row = mat.get_row_dense(0)
            >>> print(f"Max value in row 0: {max(row.tolist())}")
        """
        if i < 0 or i >= self.rows:
            raise IndexError(f"Row index {i} out of bounds [0, {self.rows})")
        
        row = zeros(self.cols, dtype=self.dtype)
        
        start = self.indptr[i]
        end = self.indptr[i + 1]
        
        for k in range(start, end):
            col_idx = self.indices[k]
            row[col_idx] = self.data[k]
        
        return row
    
    def get_row_slice(self, start: int, end: int) -> 'SclCSR':
        """
        Get a contiguous slice of rows [start:end).
        
        Convenience method for slice_rows() with range indices.
        
        Args:
            start: Starting row index (inclusive)
            end: Ending row index (exclusive)
            
        Returns:
            New SclCSR matrix with rows [start:end)
            
        Example:
            >>> # Get first 100 rows
            >>> sub_mat = mat.get_row_slice(0, 100)
            >>> 
            >>> # Get rows 50-150
            >>> sub_mat = mat.get_row_slice(50, 150)
        """
        if start < 0 or start > self.rows:
            raise IndexError(f"Start index {start} out of bounds [0, {self.rows}]")
        if end < 0 or end > self.rows:
            raise IndexError(f"End index {end} out of bounds [0, {self.rows}]")
        if start >= end:
            raise ValueError(f"Start {start} must be less than end {end}")
        
        # Create row indices array
        row_indices = from_list(list(range(start, end)), dtype='int64')
        return self.slice_rows(row_indices)
    
    def slice_rows(self, row_indices: Union[List[int], Array]) -> 'SclCSR':
        """
        Extract subset of rows (high-performance C++ implementation).
        
        Uses Inspect-Allocate-Materialize pattern:
        1. Inspect: Query output size from C++
        2. Allocate: Create output arrays
        3. Materialize: Bulk copy using memcpy in C++
        
        Args:
            row_indices: Indices of rows to keep (must be sorted)
            
        Returns:
            New SclCSR matrix with selected rows
        
        Example:
            >>> # Select first 100 cells
            >>> sub_mat = mat.slice_rows(list(range(100)))
            >>> 
            >>> # Select specific cells
            >>> keep = Array.from_list([0, 10, 20, 30], dtype='int64')
            >>> sub_mat = mat.slice_rows(keep)
        """
        if isinstance(row_indices, list):
            row_indices = from_list(row_indices, dtype='int64')
        
        if row_indices.dtype != 'int64':
            raise TypeError(f"row_indices must be int64, got {row_indices.dtype}")
        
        new_rows = len(row_indices)
        
        # Step 1: Inspect - Query output nnz from C++
        out_nnz = kernel_utils.inspect_slice_rows(
            self.indptr.get_pointer(),
            row_indices.get_pointer(),
            new_rows
        )
        
        # Step 2: Allocate output arrays
        new_data = empty(out_nnz, dtype=self.dtype)
        new_indices = empty(out_nnz, dtype='int64')
        new_indptr = zeros(new_rows + 1, dtype='int64')
        
        # Step 3: Materialize - Bulk copy in C++ (uses memcpy)
        kernel_utils.materialize_slice_rows(
            self.data.get_pointer(),
            self.indices.get_pointer(),
            self.indptr.get_pointer(),
            row_indices.get_pointer(),
            new_rows,
            new_data.get_pointer(),
            new_indices.get_pointer(),
            new_indptr.get_pointer()
        )
        
        return SclCSR(new_data, new_indices, new_indptr, shape=(new_rows, self.cols))
    
    def slice_cols(self, col_indices: Union[List[int], Array]) -> 'SclCSR':
        """
        Extract subset of columns (high-performance C++ implementation).
        
        Uses mask-based filtering in C++ for optimal performance.
        
        Args:
            col_indices: Indices of columns to keep (need not be sorted)
            
        Returns:
            New SclCSR matrix with selected columns
        
        Example:
            >>> # Select first 1000 genes
            >>> sub_mat = mat.slice_cols(list(range(1000)))
        """
        if isinstance(col_indices, list):
            col_indices = from_list(col_indices, dtype='int64')
        
        if col_indices.dtype != 'int64':
            raise TypeError(f"col_indices must be int64, got {col_indices.dtype}")
        
        new_cols = len(col_indices)
        
        # Build column mask and mapping
        col_mask = zeros(self.cols, dtype='uint8')
        col_mapping = zeros(self.cols, dtype='int64')
        
        for new_idx, old_idx in enumerate(col_indices.tolist()):
            if old_idx < 0 or old_idx >= self.cols:
                raise IndexError(f"Column index {old_idx} out of bounds")
            col_mask[old_idx] = 1
            col_mapping[old_idx] = new_idx
        
        # Step 1: Inspect - Query output nnz from C++
        out_nnz = kernel_utils.inspect_filter_cols(
            self.indices.get_pointer(),
            self.indptr.get_pointer(),
            self.rows,
            col_mask.get_pointer()
        )
        
        # Step 2: Allocate output arrays
        new_data = empty(out_nnz, dtype=self.dtype)
        new_indices = empty(out_nnz, dtype='int64')
        new_indptr = zeros(self.rows + 1, dtype='int64')
        
        # Step 3: Materialize - Filter and remap in C++
        kernel_utils.materialize_filter_cols(
            self.data.get_pointer(),
            self.indices.get_pointer(),
            self.indptr.get_pointer(),
            self.rows,
            col_mask.get_pointer(),
            col_mapping.get_pointer(),
            new_data.get_pointer(),
            new_indices.get_pointer(),
            new_indptr.get_pointer()
        )
        
        return SclCSR(new_data, new_indices, new_indptr, shape=(self.rows, new_cols))
    
    # =========================================================================
    # Alignment and Reordering
    # =========================================================================
    
    def permute_rows(self, perm: Union[List[int], Array]) -> 'SclCSR':
        """
        Permute rows according to given mapping.
        
        Args:
            perm: Permutation array [rows], where new_row[i] = old_row[perm[i]]
            
        Returns:
            New SclCSR matrix with permuted rows
        
        Example:
            >>> # Reverse row order
            >>> perm = list(range(mat.rows-1, -1, -1))
            >>> reversed_mat = mat.permute_rows(perm)
        """
        if isinstance(perm, list):
            perm = from_list(perm, dtype='int64')
        
        if len(perm) != self.rows:
            raise ValueError(f"perm size mismatch: expected {self.rows}, got {len(perm)}")
        
        # This is essentially slice_rows with a permutation
        return self.slice_rows(perm)
    
    def align_rows(
        self,
        old_to_new_map: Union[List[int], Array],
        new_rows: Optional[int] = None
    ) -> 'SclCSR':
        """
        Align rows according to mapping (supports drop and pad).
        
        **High-Performance**: Delegates to optimized C++ kernel with:
        - Two-pass algorithm (count + copy)
        - Bulk memcpy for contiguous blocks
        - Parallel processing where possible
        
        Args:
            old_to_new_map: Mapping [old_rows], where:
                            - map[i] = j means old_row[i] -> new_row[j]
                            - map[i] = -1 means drop old_row[i]
            new_rows: Optional new row count (auto-detected if None)
            
        Returns:
            New SclCSR matrix with aligned rows
        
        Example:
            >>> # Map cells to new order, drop some cells
            >>> mapping = Array.from_list([0, -1, 1, 2, -1, 3], dtype='int64')
            >>> # Old cells 1,4 are dropped; cells 0,2,3,5 -> 0,1,2,3
            >>> aligned = mat.align_rows(mapping)
        """
        if isinstance(old_to_new_map, list):
            old_to_new_map = from_list(old_to_new_map, dtype='int64')
        
        if len(old_to_new_map) != self.rows:
            raise ValueError(f"map size mismatch: expected {self.rows}, got {len(old_to_new_map)}")
        
        # Detect new_rows if not provided
        if new_rows is None:
            new_rows = max(old_to_new_map.tolist()) + 1
        
        # Count output nnz (quick scan in Python, acceptable)
        new_nnz = sum(
            self.row_lengths[i] 
            for i in range(self.rows) 
            if old_to_new_map[i] >= 0
        )
        
        # Allocate output arrays
        new_data = empty(new_nnz, dtype=self.dtype)
        new_indices = empty(new_nnz, dtype='int64')
        new_indptr = zeros(new_rows + 1, dtype='int64')
        new_row_lengths = zeros(new_rows, dtype='int64')
        
        # Delegate to C++ kernel (two-pass algorithm with bulk copy)
        kernel_utils.align_rows(
            self.data.get_pointer(),
            self.indices.get_pointer(),
            self.indptr.get_pointer(),
            self.rows,
            old_to_new_map.get_pointer(),
            new_rows,
            new_data.get_pointer(),
            new_indices.get_pointer(),
            new_indptr.get_pointer(),
            new_row_lengths.get_pointer()
        )
        
        return SclCSR(new_data, new_indices, new_indptr, shape=(new_rows, self.cols),
                     row_lengths=new_row_lengths)
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def __repr__(self) -> str:
        return (f"SclCSR(shape={self.shape}, nnz={self.nnz}, "
                f"dtype={self.data.dtype})")
    
    def get_c_pointers(self) -> Tuple:
        """
        Get C-compatible pointers for kernel calls.
        
        Returns:
            (data_ptr, indices_ptr, indptr_ptr, row_lengths_ptr, rows, cols)
        """
        return (
            self.data.get_pointer(),
            self.indices.get_pointer(),
            self.indptr.get_pointer(),
            self.row_lengths.get_pointer(),
            self.rows,
            self.cols
        )


class SclCSC:
    """
    SCL Custom CSC Matrix (Cells × Genes, column-major).
    
    Similar to SclCSR but column-oriented for efficient gene-wise operations.
    """
    
    def __init__(
        self,
        data: Array,
        indices: Array,
        indptr: Array,
        shape: Tuple[int, int],
        col_lengths: Optional[Array] = None
    ):
        """Initialize SclCSC matrix."""
        # Type validation
        if data.dtype not in ('float32', 'float64'):
            raise TypeError(f"data must be float32/float64, got {data.dtype}")
        if indices.dtype != 'int64':
            raise TypeError(f"indices must be int64, got {indices.dtype}")
        if indptr.dtype != 'int64':
            raise TypeError(f"indptr must be int64, got {indptr.dtype}")
        
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = tuple(shape)
        
        # Validate dimensions
        rows, cols = self.shape
        if rows < 0 or cols < 0:
            raise ValueError(f"Invalid shape: {self.shape}")
        
        if len(self.indptr) != cols + 1:
            raise ValueError(f"indptr size mismatch: expected {cols+1}, got {len(self.indptr)}")
        
        nnz = len(self.data)
        if len(self.indices) != nnz:
            raise ValueError(f"indices size mismatch: expected {nnz}, got {len(self.indices)}")
        
        # Compute or validate col_lengths
        if col_lengths is None:
            self.col_lengths = self._compute_col_lengths()
        else:
            if col_lengths.dtype != 'int64':
                raise TypeError(f"col_lengths must be int64, got {col_lengths.dtype}")
            if len(col_lengths) != cols:
                raise ValueError(f"col_lengths size mismatch: expected {cols}, got {len(col_lengths)}")
            self.col_lengths = col_lengths
    
    def _compute_col_lengths(self) -> Array:
        """
        Compute column lengths from indptr (delegated to C++).
        
        This uses parallel diff operation in C++, ~100x faster than Python loop.
        """
        cols = self.shape[1]
        lengths = zeros(cols, dtype='int64')
        
        # Call C++ kernel: parallel diff operation
        kernel_utils.compute_lengths(
            self.indptr.get_pointer(),
            cols,
            lengths.get_pointer()
        )
        
        return lengths
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.data)
    
    @property
    def rows(self) -> int:
        """Number of rows."""
        return self.shape[0]
    
    @property
    def cols(self) -> int:
        """Number of columns."""
        return self.shape[1]
    
    @property
    def dtype(self) -> str:
        """Data type."""
        return self.data.dtype
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def empty(cls, rows: int, cols: int, nnz: int, dtype: str = 'float64') -> 'SclCSC':
        """Create empty CSC matrix with pre-allocated arrays."""
        data = empty(nnz, dtype=dtype)
        indices = empty(nnz, dtype='int64')
        indptr = zeros(cols + 1, dtype='int64')
        
        return cls(data, indices, indptr, shape=(rows, cols))
    
    @classmethod
    def zeros(cls, rows: int, cols: int, dtype: str = 'float64') -> 'SclCSC':
        """Create zero matrix."""
        data = empty(0, dtype=dtype)
        indices = empty(0, dtype='int64')
        indptr = zeros(cols + 1, dtype='int64')
        
        return cls(data, indices, indptr, shape=(rows, cols))
    
    @classmethod
    def from_scipy(cls, mat: Any) -> 'SclCSC':
        """Create SclCSC from scipy.sparse.csc_matrix."""
        try:
            import scipy.sparse as sp
        except ImportError:
            raise ImportError("scipy is required for from_scipy()")
        
        if not sp.isspmatrix_csc(mat):
            raise TypeError("Input must be scipy.sparse.csc_matrix")
        
        mat.sort_indices()
        mat.eliminate_zeros()
        
        import numpy as np
        
        if mat.data.dtype == np.float32:
            dtype = 'float32'
        else:
            dtype = 'float64'
        
        # Ensure int64 for indices (required by C API)
        if mat.indices.dtype != np.int64:
            mat.indices = mat.indices.astype(np.int64)
        if mat.indptr.dtype != np.int64:
            mat.indptr = mat.indptr.astype(np.int64)
        
        data = Array.from_buffer(mat.data, dtype, len(mat.data))
        indices = Array.from_buffer(mat.indices, 'int64', len(mat.indices))
        indptr = Array.from_buffer(mat.indptr, 'int64', len(mat.indptr))
        
        return cls(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=mat.shape
        )
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_scipy(self) -> Any:
        """Convert to scipy.sparse.csc_matrix."""
        try:
            import scipy.sparse as sp
            import numpy as np
        except ImportError:
            raise ImportError("scipy and numpy are required for to_scipy()")
        
        np_dtype = np.float32 if self.data.dtype == 'float32' else np.float64
        
        data_np = np.frombuffer(self.data.tobytes(), dtype=np_dtype)
        indices_np = np.frombuffer(self.indices.tobytes(), dtype=np.int64)
        indptr_np = np.frombuffer(self.indptr.tobytes(), dtype=np.int64)
        
        return sp.csc_matrix(
            (data_np, indices_np, indptr_np),
            shape=self.shape
        )
    
    def tocsr(self) -> SclCSR:
        """Convert to CSR format."""
        scipy_csc = self.to_scipy()
        scipy_csr = scipy_csc.tocsr()
        return SclCSR.from_scipy(scipy_csr)
    
    def copy(self) -> 'SclCSC':
        """Create a deep copy."""
        return SclCSC(
            data=self.data.copy(),
            indices=self.indices.copy(),
            indptr=self.indptr.copy(),
            shape=self.shape,
            col_lengths=self.col_lengths.copy()
        )
    
    # =========================================================================
    # Slicing Operations
    # =========================================================================
    
    def get_col(self, j: int) -> Tuple[Array, Array]:
        """
        Get column j as sparse representation (indices, values).
        
        Args:
            j: Column index
            
        Returns:
            (indices, values) - Row indices and values for non-zeros
        """
        if j < 0 or j >= self.cols:
            raise IndexError(f"Column index {j} out of bounds [0, {self.cols})")
        
        start = self.indptr[j]
        end = self.indptr[j + 1]
        length = end - start
        
        if length == 0:
            return empty(0, dtype='int64'), empty(0, dtype=self.dtype)
        
        col_indices = empty(length, dtype='int64')
        col_values = empty(length, dtype=self.dtype)
        
        for k in range(length):
            col_indices[k] = self.indices[start + k]
            col_values[k] = self.data[start + k]
        
        return col_indices, col_values
    
    def get_col_dense(self, j: int) -> Array:
        """
        Get column j as dense array.
        
        Args:
            j: Column index
            
        Returns:
            Dense column vector [rows]
        """
        if j < 0 or j >= self.cols:
            raise IndexError(f"Column index {j} out of bounds [0, {self.cols})")
        
        col = zeros(self.rows, dtype=self.dtype)
        
        start = self.indptr[j]
        end = self.indptr[j + 1]
        
        for k in range(start, end):
            row_idx = self.indices[k]
            col[row_idx] = self.data[k]
        
        return col
    
    def slice_cols(self, col_indices: Union[List[int], Array]) -> 'SclCSC':
        """
        Extract subset of columns.
        
        Args:
            col_indices: Indices of columns to keep (must be sorted)
            
        Returns:
            New SclCSC matrix with selected columns
        """
        if isinstance(col_indices, list):
            col_indices = from_list(col_indices, dtype='int64')
        
        if col_indices.dtype != 'int64':
            raise TypeError(f"col_indices must be int64, got {col_indices.dtype}")
        
        new_cols = len(col_indices)
        
        # Count total nnz
        total_nnz = 0
        for j in range(new_cols):
            col_idx = col_indices[j]
            if col_idx < 0 or col_idx >= self.cols:
                raise IndexError(f"Column index {col_idx} out of bounds")
            total_nnz += self.col_lengths[col_idx]
        
        # Allocate new arrays
        new_data = empty(total_nnz, dtype=self.dtype)
        new_indices = empty(total_nnz, dtype='int64')
        new_indptr = zeros(new_cols + 1, dtype='int64')
        
        # Copy data
        nnz_pos = 0
        for j in range(new_cols):
            col_idx = col_indices[j]
            start = self.indptr[col_idx]
            end = self.indptr[col_idx + 1]
            length = end - start
            
            for k in range(length):
                new_data[nnz_pos] = self.data[start + k]
                new_indices[nnz_pos] = self.indices[start + k]
                nnz_pos += 1
            
            new_indptr[j + 1] = nnz_pos
        
        return SclCSC(new_data, new_indices, new_indptr, shape=(self.rows, new_cols))
    
    def align_cols(
        self,
        old_to_new_map: Union[List[int], Array],
        new_cols: Optional[int] = None
    ) -> 'SclCSC':
        """
        Align columns according to mapping (supports drop and pad).
        
        **High-Performance**: Delegates to optimized C++ kernel.
        Same algorithm as align_rows but for columns.
        
        Args:
            old_to_new_map: Mapping [old_cols], where:
                           - map[j] = k means old_col[j] -> new_col[k]
                           - map[j] = -1 means drop old_col[j]
            new_cols: Optional new column count (auto-detected if None)
            
        Returns:
            New SclCSC matrix with aligned columns
        """
        if isinstance(old_to_new_map, list):
            old_to_new_map = from_list(old_to_new_map, dtype='int64')
        
        if len(old_to_new_map) != self.cols:
            raise ValueError(f"map size mismatch: expected {self.cols}, got {len(old_to_new_map)}")
        
        if new_cols is None:
            new_cols = max(old_to_new_map.tolist()) + 1
        
        # Count output nnz
        new_nnz = sum(
            self.col_lengths[j]
            for j in range(self.cols)
            if old_to_new_map[j] >= 0
        )
        
        # Allocate output arrays
        new_data = empty(new_nnz, dtype=self.dtype)
        new_indices = empty(new_nnz, dtype='int64')
        new_indptr = zeros(new_cols + 1, dtype='int64')
        new_col_lengths = zeros(new_cols, dtype='int64')
        
        # Delegate to C++ kernel (reuse align_rows, works for CSC too)
        kernel_utils.align_rows(
            self.data.get_pointer(),
            self.indices.get_pointer(),
            self.indptr.get_pointer(),
            self.cols,  # CSC: iterate over columns
            old_to_new_map.get_pointer(),
            new_cols,
            new_data.get_pointer(),
            new_indices.get_pointer(),
            new_indptr.get_pointer(),
            new_col_lengths.get_pointer()
        )
        
        return SclCSC(new_data, new_indices, new_indptr, shape=(self.rows, new_cols),
                     col_lengths=new_col_lengths)
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def __repr__(self) -> str:
        return (f"SclCSC(shape={self.shape}, nnz={self.nnz}, "
                f"dtype={self.data.dtype})")
    
    def get_c_pointers(self) -> Tuple:
        """
        Get C-compatible pointers for kernel calls.
        
        Returns:
            (data_ptr, indices_ptr, indptr_ptr, col_lengths_ptr, rows, cols)
        """
        return (
            self.data.get_pointer(),
            self.indices.get_pointer(),
            self.indptr.get_pointer(),
            self.col_lengths.get_pointer(),
            self.rows,
            self.cols
        )
