"""
Sparse Matrix Base Classes

This module defines the abstract base classes for the SCL sparse matrix type system.
It establishes a unified interface that all sparse matrix implementations must follow,
enabling polymorphic operations across different storage backends.

Type Hierarchy:

    SparseBase (ABC)
    ├── CSRBase (ABC) - Row-oriented sparse matrices
    │   ├── SclCSR - Standard CSR with multiple backends
    │   └── CallbackCSR - User-defined row access via callbacks
    └── CSCBase (ABC) - Column-oriented sparse matrices
        ├── SclCSC - Standard CSC with multiple backends
        └── CallbackCSC - User-defined column access via callbacks

Design Philosophy:

1. Unified Interface: All sparse matrices share common properties (shape, dtype, nnz)
   and operations (sum, mean, etc.) regardless of storage backend.

2. Backend Agnostic: Operations work the same whether data is in memory, memory-mapped,
   or accessed through callbacks.

3. Python Inheritance: Unlike C++ concepts which use static dispatch, Python uses
   dynamic dispatch via inheritance. This allows method overriding at runtime.

4. Interoperability: All sparse matrices can convert to/from scipy sparse matrices
   and work with standard Python scientific computing tools.

Example:

    # All these work the same way
    mat1 = SclCSR.from_scipy(scipy_mat)     # Standard CSR
    mat2 = CallbackCSR(lazy_loader)          # Callback-based
    mat3 = SclCSR.from_h5ad("data.h5ad")    # Memory-mapped
    
    for mat in [mat1, mat2, mat3]:
        row_sums = mat.sum(axis=1)  # Same interface!
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    from scipy.sparse import spmatrix
    from ._array import Array

__all__ = [
    'SparseBase',
    'CSRBase', 
    'CSCBase',
    'SparseFormat',
]


class SparseFormat:
    """Enumeration of sparse matrix formats."""
    CSR = 'csr'
    CSC = 'csc'


class SparseBase(ABC):
    """
    Abstract base class for all sparse matrices.
    
    This class defines the minimal interface that all sparse matrix
    implementations must provide. It enables polymorphic operations
    across different storage backends (Custom, Virtual, Mapped, Callback).
    
    Required Properties (subclasses must implement):
        shape: Matrix dimensions (rows, cols)
        dtype: Data type string ('float32', 'float64')
        nnz: Number of non-zero elements
        format: Sparse format ('csr' or 'csc')
    
    Optional Properties (subclasses may override):
        is_view: Whether this is a view of another matrix
        is_contiguous: Whether data is contiguous in memory
    
    Required Methods (subclasses must implement):
        sum(axis): Compute sums along axis
        mean(axis): Compute means along axis
        to_scipy(): Convert to scipy sparse matrix
        copy(): Create a deep copy
    
    Example:
    
        class MySparseMatrix(SparseBase):
            @property
            def shape(self) -> Tuple[int, int]:
                return self._shape
            
            @property
            def dtype(self) -> str:
                return self._dtype
            
            # ... implement other required methods
    """
    
    # =========================================================================
    # Abstract Properties
    # =========================================================================
    
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Matrix dimensions (rows, cols)."""
        ...
    
    @property
    @abstractmethod
    def dtype(self) -> str:
        """Data type string ('float32' or 'float64')."""
        ...
    
    @property
    @abstractmethod
    def nnz(self) -> int:
        """Number of non-zero elements."""
        ...
    
    @property
    @abstractmethod
    def format(self) -> str:
        """Sparse format ('csr' or 'csc')."""
        ...
    
    # =========================================================================
    # Derived Properties
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
    def ndim(self) -> int:
        """Number of dimensions (always 2 for sparse matrices)."""
        return 2
    
    @property
    def size(self) -> int:
        """Total number of elements (rows * cols)."""
        return self.shape[0] * self.shape[1]
    
    @property
    def density(self) -> float:
        """Fraction of non-zero elements."""
        total = self.size
        return self.nnz / total if total > 0 else 0.0
    
    @property
    def is_view(self) -> bool:
        """Whether this matrix is a view of another matrix.
        
        Default is False. Subclasses may override.
        """
        return False
    
    @property
    def is_contiguous(self) -> bool:
        """Whether data is contiguous in memory.
        
        Default is True. Subclasses may override.
        """
        return True
    
    # =========================================================================
    # Abstract Methods - Statistics
    # =========================================================================
    
    @abstractmethod
    def sum(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute sum along axis.
        
        Args:
            axis: None for total sum, 0 for column sums, 1 for row sums
            
        Returns:
            Scalar (axis=None) or 1D array (axis=0 or 1)
        """
        ...
    
    @abstractmethod
    def mean(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute mean along axis.
        
        Args:
            axis: None for total mean, 0 for column means, 1 for row means
            
        Returns:
            Scalar (axis=None) or 1D array (axis=0 or 1)
        """
        ...
    
    # =========================================================================
    # Abstract Methods - Conversion
    # =========================================================================
    
    @abstractmethod
    def to_scipy(self) -> 'spmatrix':
        """Convert to scipy sparse matrix.
        
        Returns:
            scipy.sparse.csr_matrix or scipy.sparse.csc_matrix
        """
        ...
    
    @abstractmethod
    def copy(self) -> 'SparseBase':
        """Create a deep copy of this matrix.
        
        Returns:
            New matrix with owned data
        """
        ...
    
    # =========================================================================
    # Optional Methods - Subclasses may override for optimization
    # =========================================================================
    
    def min(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute minimum along axis.
        
        Default implementation converts to scipy. Subclasses may override.
        """
        scipy_mat = self.to_scipy()
        if axis is None:
            return scipy_mat.min()
        return np.asarray(scipy_mat.min(axis=axis)).ravel()
    
    def max(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute maximum along axis.
        
        Default implementation converts to scipy. Subclasses may override.
        """
        scipy_mat = self.to_scipy()
        if axis is None:
            return scipy_mat.max()
        return np.asarray(scipy_mat.max(axis=axis)).ravel()
    
    def var(self, axis: Optional[int] = None, ddof: int = 0) -> Union[float, np.ndarray]:
        """Compute variance along axis.
        
        Default implementation uses sum/mean. Subclasses may override.
        """
        # Var(X) = E[X^2] - E[X]^2
        mean_val = self.mean(axis=axis)
        # This is a fallback - subclasses should implement efficiently
        scipy_mat = self.to_scipy()
        sq_mat = scipy_mat.power(2)
        if axis is None:
            sq_mean = sq_mat.sum() / self.size
            return sq_mean - mean_val ** 2
        else:
            n = self.shape[1 - axis]
            sq_mean = np.asarray(sq_mat.sum(axis=axis)).ravel() / n
            return sq_mean - mean_val ** 2
    
    def std(self, axis: Optional[int] = None, ddof: int = 0) -> Union[float, np.ndarray]:
        """Compute standard deviation along axis.
        
        Default implementation uses var(). Subclasses may override.
        """
        return np.sqrt(self.var(axis=axis, ddof=ddof))
    
    def to_dense(self) -> np.ndarray:
        """Convert to dense numpy array.
        
        Default implementation converts to scipy first.
        """
        return self.to_scipy().toarray()
    
    # =========================================================================
    # Magic Methods
    # =========================================================================
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"shape={self.shape}, nnz={self.nnz}, "
                f"dtype={self.dtype}, format={self.format})")
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __len__(self) -> int:
        """Return number of rows."""
        return self.shape[0]
    
    def __bool__(self) -> bool:
        """Return True if matrix has any elements."""
        return self.nnz > 0


class CSRBase(SparseBase):
    """
    Abstract base class for CSR (Compressed Sparse Row) matrices.
    
    Extends SparseBase with CSR-specific properties and methods.
    CSR format is optimal for:
    - Row slicing and iteration
    - Matrix-vector products (Ax)
    - Row-wise statistics (per-cell metrics in scRNA-seq)
    
    Additional Required Methods:
        row_values(i): Get values for row i
        row_indices(i): Get column indices for row i
        row_length(i): Get number of non-zeros in row i
    
    Properties:
        format: Always returns 'csr'
    """
    
    @property
    def format(self) -> str:
        """Sparse format (always 'csr')."""
        return SparseFormat.CSR
    
    # =========================================================================
    # Abstract Methods - Row Access
    # =========================================================================
    
    @abstractmethod
    def row_values(self, i: int) -> 'Array':
        """Get non-zero values for row i.
        
        Args:
            i: Row index
            
        Returns:
            Array of non-zero values in row i
        """
        ...
    
    @abstractmethod
    def row_indices(self, i: int) -> 'Array':
        """Get column indices of non-zeros for row i.
        
        Args:
            i: Row index
            
        Returns:
            Array of column indices for non-zeros in row i
        """
        ...
    
    @abstractmethod
    def row_length(self, i: int) -> int:
        """Get number of non-zeros in row i.
        
        Args:
            i: Row index
            
        Returns:
            Number of non-zero elements in row i
        """
        ...
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def get_row(self, i: int) -> Tuple['Array', 'Array']:
        """Get both values and indices for row i.
        
        Args:
            i: Row index
            
        Returns:
            Tuple of (values, indices) arrays
        """
        return self.row_values(i), self.row_indices(i)
    
    def iter_rows(self):
        """Iterate over rows, yielding (values, indices) tuples.
        
        Yields:
            Tuple of (values, indices) for each row
        """
        for i in range(self.rows):
            yield self.row_values(i), self.row_indices(i)


class CSCBase(SparseBase):
    """
    Abstract base class for CSC (Compressed Sparse Column) matrices.
    
    Extends SparseBase with CSC-specific properties and methods.
    CSC format is optimal for:
    - Column slicing and iteration
    - Transposed matrix-vector products (A^T x)
    - Column-wise statistics (per-gene metrics in scRNA-seq)
    
    Additional Required Methods:
        col_values(j): Get values for column j
        col_indices(j): Get row indices for column j
        col_length(j): Get number of non-zeros in column j
    
    Properties:
        format: Always returns 'csc'
    """
    
    @property
    def format(self) -> str:
        """Sparse format (always 'csc')."""
        return SparseFormat.CSC
    
    # =========================================================================
    # Abstract Methods - Column Access
    # =========================================================================
    
    @abstractmethod
    def col_values(self, j: int) -> 'Array':
        """Get non-zero values for column j.
        
        Args:
            j: Column index
            
        Returns:
            Array of non-zero values in column j
        """
        ...
    
    @abstractmethod
    def col_indices(self, j: int) -> 'Array':
        """Get row indices of non-zeros for column j.
        
        Args:
            j: Column index
            
        Returns:
            Array of row indices for non-zeros in column j
        """
        ...
    
    @abstractmethod
    def col_length(self, j: int) -> int:
        """Get number of non-zeros in column j.
        
        Args:
            j: Column index
            
        Returns:
            Number of non-zero elements in column j
        """
        ...
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def get_col(self, j: int) -> Tuple['Array', 'Array']:
        """Get both values and indices for column j.
        
        Args:
            j: Column index
            
        Returns:
            Tuple of (values, indices) arrays
        """
        return self.col_values(j), self.col_indices(j)
    
    def iter_cols(self):
        """Iterate over columns, yielding (values, indices) tuples.
        
        Yields:
            Tuple of (values, indices) for each column
        """
        for j in range(self.cols):
            yield self.col_values(j), self.col_indices(j)

