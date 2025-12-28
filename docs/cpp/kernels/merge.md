# merge.hpp

> scl/kernel/merge.hpp · Matrix merging and concatenation operations

## Overview

Efficient sparse matrix concatenation operations for combining matrices along primary (rows) or secondary (columns) dimensions. These operations are essential for data integration, batch merging, and feature combination in single-cell analysis.

This file provides:
- Vertical stacking (vstack) - concatenate along rows
- Horizontal stacking (hstack) - concatenate along columns
- SIMD-optimized index offset operations
- Parallel memory copy for large matrices

**Header**: `#include "scl/kernel/merge.hpp"`

---

## Main APIs

### vstack

::: source_code file="scl/kernel/merge.hpp" symbol="vstack" collapsed
:::

**Algorithm Description**

Vertically stacks two sparse matrices by concatenating along the primary dimension (rows for CSR, columns for CSC):

1. **Dimension validation**: Check that matrices can be vertically stacked (secondary dimensions can differ)
2. **Result size calculation**:
   - Result primary_dim = matrix1.primary_dim + matrix2.primary_dim
   - Result secondary_dim = max(matrix1.secondary_dim, matrix2.secondary_dim)
   - Result nnz = matrix1.nnz + matrix2.nnz
3. **Memory allocation**: Allocate result matrix using specified block strategy
4. **Data copying** (parallel over rows):
   - Copy matrix1 rows to result[0 : n1] (indices unchanged)
   - Copy matrix2 rows to result[n1 : n1+n2] (indices unchanged, may have gaps if secondary_dim differs)
5. **Index pointer setup**: Set indptr array correctly for combined structure

For CSR matrices: Rows from matrix1 are placed first, followed by rows from matrix2. Column indices remain unchanged.

**Edge Cases**

- **Empty matrix1**: Returns copy of matrix2
- **Empty matrix2**: Returns copy of matrix1
- **Both empty**: Returns empty matrix with correct dimensions
- **Different secondary dimensions**: Uses max dimension, matrix2 indices remain valid (no offset needed)
- **One matrix has zero secondary_dim**: Handled correctly, result uses non-zero dimension

**Data Guarantees (Preconditions)**

- Both matrices must be valid sparse matrices
- Matrices must have same format (both CSR or both CSC)
- For CSR: Secondary dimensions (columns) can differ
- For CSC: Secondary dimensions (rows) can differ
- Block strategy is used for result allocation

**Complexity Analysis**

- **Time**: O(nnz1 + nnz2) where nnz1 and nnz2 are the number of non-zeros in each matrix. Parallel copy operations reduce effective time.
- **Space**: O(nnz1 + nnz2) for the result matrix storage

**Example**

```cpp
#include "scl/kernel/merge.hpp"

Sparse<Real, true> matrix1 = /* n1 x m1 sparse matrix */;
Sparse<Real, true> matrix2 = /* n2 x m2 sparse matrix */;

// Vertical stacking (for CSR: stack rows)
auto vstacked = scl::kernel::merge::vstack(matrix1, matrix2);

// Result is (n1+n2) x max(m1, m2) sparse matrix
// Rows 0 to n1-1 come from matrix1
// Rows n1 to n1+n2-1 come from matrix2

// With custom block allocation strategy
auto vstacked_custom = scl::kernel::merge::vstack(
    matrix1, matrix2, BlockStrategy::adaptive()
);
```

---

### hstack

::: source_code file="scl/kernel/merge.hpp" symbol="hstack" collapsed
:::

**Algorithm Description**

Horizontally stacks two sparse matrices by concatenating along the secondary dimension (columns for CSR, rows for CSC):

1. **Dimension validation**: Verify that primary dimensions match (rows for CSR, columns for CSC)
2. **Result size calculation**:
   - Result primary_dim = matrix1.primary_dim (unchanged)
   - Result secondary_dim = matrix1.secondary_dim + matrix2.secondary_dim
   - Result nnz = matrix1.nnz + matrix2.nnz
3. **Memory allocation**: Allocate result matrix
4. **Data copying** (parallel over rows for CSR):
   - For each row in parallel:
     - Copy matrix1 values and indices to result
     - Copy matrix2 values to result
     - Add offset (matrix1.secondary_dim) to matrix2 indices using SIMD-optimized addition
     - Merge sorted indices if needed (both matrices should have sorted indices)
5. **Index merging**: Combine indices from both matrices per row, maintaining sorted order

For CSR matrices: Each row contains columns from matrix1 followed by columns from matrix2. Matrix2 column indices are offset by matrix1.cols().

**Edge Cases**

- **Empty matrix1**: Returns matrix2 with indices unchanged (no offset needed)
- **Empty matrix2**: Returns copy of matrix1
- **Both empty**: Returns empty matrix with correct dimensions
- **Mismatched primary dimensions**: Throws DimensionError
- **Zero offset**: Early exit optimization (no index adjustment needed)

**Data Guarantees (Preconditions)**

- Both matrices must be valid sparse matrices
- Matrices must have same format (both CSR or both CSC)
- Primary dimensions must match: matrix1.primary_dim == matrix2.primary_dim
- Indices should be sorted within rows (CSR) or columns (CSC) for optimal performance
- Block strategy is used for result allocation

**Complexity Analysis**

- **Time**: O(nnz1 + nnz2) for data copying. Index offset addition uses SIMD optimization for bulk operations. Index merging (if needed) adds O(n * log(k)) where n is number of rows and k is average non-zeros per row.
- **Space**: O(nnz1 + nnz2) for the result matrix

**Example**

```cpp
#include "scl/kernel/merge.hpp"

Sparse<Real, true> matrix1 = /* n x m1 sparse matrix */;
Sparse<Real, true> matrix2 = /* n x m2 sparse matrix */;  // Same number of rows

// Horizontal stacking (for CSR: stack columns)
auto hstacked = scl::kernel::merge::hstack(matrix1, matrix2);

// Result is n x (m1+m2) sparse matrix
// Each row contains columns 0 to m1-1 from matrix1,
// followed by columns m1 to m1+m2-1 from matrix2
// Matrix2 column indices are offset by m1

// With custom block strategy
auto hstacked_custom = scl::kernel::merge::hstack(
    matrix1, matrix2, BlockStrategy::adaptive()
);
```

---

## Implementation Details

### SIMD Optimization

The `hstack` function uses SIMD (Single Instruction Multiple Data) instructions for efficient index offset addition:

- When offset > 0, bulk addition uses 2-way SIMD unrolled loops
- Scalar cleanup handles remainder elements
- Early exit optimization when offset == 0 (direct memcpy)

### Parallel Processing

Both `vstack` and `hstack` use parallel processing:

- **vstack**: Parallel copy over rows/columns (primary dimension)
- **hstack**: Parallel processing over rows (CSR) or columns (CSC)
- Large data blocks use parallel memcpy with prefetching

### Memory Management

- Result matrices are allocated using `BlockStrategy` for efficient sparse storage
- Default strategy is `BlockStrategy::adaptive()` which chooses optimal block size
- Memory is allocated contiguously when possible for better cache performance

## Notes

- **Index sorting**: For optimal performance, input matrices should have sorted indices within each row (CSR) or column (CSC). The implementation maintains sorted order in results.
- **Format consistency**: Both input matrices must use the same storage format (both CSR or both CSC).
- **Dimension constraints**: 
  - `vstack`: Secondary dimensions can differ (uses maximum)
  - `hstack`: Primary dimensions must match exactly
- **Sparse efficiency**: These operations are optimized for sparse matrices and preserve sparsity structure efficiently.

## See Also

- [Sparse Matrices](../core/sparse) - Sparse matrix data structure documentation
- [Memory Management](../core/memory) - Block allocation strategies
