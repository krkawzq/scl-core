# reorder.hpp

> scl/kernel/reorder.hpp · Matrix reordering and permutation operations

## Overview

This file provides functions for reordering rows and columns of sparse matrices according to permutation vectors. Reordering is useful for clustering visualization, data organization, and optimizing memory access patterns.

Key features:
- Row and column reordering
- Preserves matrix structure (CSR/CSC format)
- Parallelized implementations
- Memory-efficient operations

**Header**: `#include "scl/kernel/reorder.hpp"`

---

## Main APIs

### reorder_rows

::: source_code file="scl/kernel/reorder.hpp" symbol="reorder_rows" collapsed
:::

**Algorithm Description**

Reorder rows of sparse matrix according to permutation:

1. **Allocate output matrix**: Create new sparse matrix with same dimensions
2. **Parallel processing**: For each output row i in parallel:
   - Source row index: `src_row = permutation[i]`
   - Copy row data: Copy all non-zero elements from `matrix[src_row]` to `output[i]`
   - Preserve column indices and values
3. **Update structure**: Set output matrix structure (indptr, indices, values)

The permutation vector maps output row indices to input row indices: `output[i] = matrix[permutation[i]]`.

**Edge Cases**

- **Identity permutation**: Output is identical to input (but new matrix allocated)
- **Empty rows**: Preserved in output (row with no non-zeros)
- **Invalid permutation**: Undefined behavior if permutation contains out-of-range indices
- **Duplicate indices**: Undefined behavior if permutation has duplicates

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR format (for row reordering)
- `permutation.len >= n_rows`
- Permutation must be valid: `permutation[i]` in range [0, matrix.rows())
- Permutation should be a bijection (each index appears exactly once)

**Complexity Analysis**

- **Time**: O(nnz) - single pass through all non-zero elements
  - Parallelized over rows
- **Space**: O(nnz) auxiliary for output matrix

**Example**

```cpp
#include "scl/kernel/reorder.hpp"

Sparse<Real, true> matrix = /* ... */;  // n_rows x n_cols
Array<Index> permutation(n_rows);
// ... set permutation: permutation[i] = source row index for output row i ...

Sparse<Real, true> output;
scl::kernel::reorder::reorder_rows(
    matrix,
    permutation,
    n_rows,
    output
);

// output[i] now contains matrix[permutation[i]]
```

---

### reorder_columns

::: source_code file="scl/kernel/reorder.hpp" symbol="reorder_columns" collapsed
:::

**Algorithm Description**

Reorder columns of sparse matrix according to permutation:

1. **Allocate output matrix**: Create new sparse matrix with same dimensions
2. **Parallel processing**: For each row in parallel:
   - For each non-zero element in row:
     - Original column: `src_col = indices[j]`
     - New column: `dst_col = permutation[src_col]`
     - Copy value to output at `(row, dst_col)`
3. **Sort columns**: Sort column indices within each row (CSR requirement)

Column reordering requires remapping column indices: `output[i, permutation[j]] = matrix[i, j]`.

**Edge Cases**

- **Identity permutation**: Output is identical to input
- **Empty columns**: Columns with no non-zeros are preserved
- **Invalid permutation**: Undefined behavior if permutation contains out-of-range indices
- **Duplicate indices**: Undefined behavior if permutation has duplicates

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR or CSC format
- `permutation.len >= n_cols`
- Permutation must be valid: `permutation[i]` in range [0, matrix.cols())
- Permutation should be a bijection (each index appears exactly once)

**Complexity Analysis**

- **Time**: O(nnz) - single pass through all non-zero elements
  - Additional O(nnz * log(nnz_per_row)) for sorting column indices within rows
  - Parallelized over rows
- **Space**: O(nnz) auxiliary for output matrix

**Example**

```cpp
Sparse<Real, true> matrix = /* ... */;
Array<Index> permutation(n_cols);
// ... set permutation: permutation[i] = source column index for output column i ...

Sparse<Real, true> output;
scl::kernel::reorder::reorder_columns(
    matrix,
    permutation,
    n_cols,
    output
);

// output[i, permutation[j]] now contains matrix[i, j]
```

---

## Notes

**Permutation Validity**

A valid permutation must:
- Contain all indices in range [0, n) exactly once
- Be a bijection (one-to-one mapping)

**Use Cases**

- **Clustering visualization**: Reorder rows/columns to group similar items
- **Data organization**: Sort by metadata (e.g., cell type, batch)
- **Memory optimization**: Reorder for better cache locality
- **Matrix operations**: Prepare matrices for efficient algorithms

**Thread Safety**

Both functions are thread-safe and parallelized:
- `reorder_rows`: Parallel over output rows
- `reorder_columns`: Parallel over input rows

## See Also

- [Slice](/cpp/kernels/slice) - Matrix slicing operations
- [Merge](/cpp/kernels/merge) - Matrix merging operations
