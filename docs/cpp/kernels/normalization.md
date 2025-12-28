# normalize.hpp

> scl/kernel/normalize.hpp · Normalization operations with SIMD optimization

## Overview

This file provides high-performance normalization operations for sparse matrices with SIMD acceleration and efficient parallel processing.

This file provides:
- Row/column sum computation
- Primary dimension scaling
- Masked sum operations
- Highly expressed gene detection

**Header**: `#include "scl/kernel/normalize.hpp"`

---

## Main APIs

### compute_row_sums

::: source_code file="scl/kernel/normalize.hpp" symbol="compute_row_sums" collapsed
:::

**Algorithm Description**

Compute sum of values in each row of a sparse matrix:

1. **Parallel Processing**: Process each row in parallel:
   - Extract row values from sparse matrix
   - Iterate over non-zero elements in row

2. **Vectorized Summation**: For each row:
   - Use SIMD operations to sum values efficiently
   - Handle rows of different lengths optimally
   - Accumulate sum into output buffer

3. **Output**: Store row sums in pre-allocated buffer:
   - `output[i]` = sum of all values in row i

**Edge Cases**

- **Empty rows**: Rows with no non-zeros get sum = 0
- **Zero values**: Zero values are included in sum (though sparse format may not store them)
- **NaN/Inf**: Propagates through standard floating-point arithmetic
- **Very sparse rows**: Handled efficiently with minimal overhead

**Data Guarantees (Preconditions)**

- `output.len >= matrix.rows()`
- Matrix must be valid sparse format (CSR or CSC)
- Output buffer must be pre-allocated

**Complexity Analysis**

- **Time**: O(nnz) where nnz is number of non-zeros
  - Each non-zero is accessed once
  - Parallelized over rows
- **Space**: O(1) auxiliary space per thread

**Example**

```cpp
#include "scl/kernel/normalize.hpp"

scl::Sparse<Real, true> matrix = /* ... */;  // [n_rows x n_cols]
scl::Array<Real> row_sums(matrix.rows());

scl::kernel::normalize::compute_row_sums(matrix, row_sums);

// row_sums[i] contains sum of row i
// Use for normalization: normalized_value = value / row_sums[row_idx]
```

---

### scale_primary

::: source_code file="scl/kernel/normalize.hpp" symbol="scale_primary" collapsed
:::

**Algorithm Description**

Scale each primary dimension (row for CSR, column for CSC) by a factor:

1. **Early Exit Optimization**: For each primary dimension:
   - If scale factor == 1.0, skip (no-op)
   - Avoids unnecessary memory access

2. **SIMD Scaling**: For each primary dimension:
   - Load values for this dimension
   - Scale using SIMD operations (4-way or 8-way unroll)
   - Store scaled values back to matrix

3. **In-place Update**: Modify matrix.values() directly:
   - No temporary allocations
   - Preserves matrix structure (indices, indptr unchanged)

**Edge Cases**

- **Unit scaling**: Scale factor = 1.0 is optimized (early exit)
- **Zero scaling**: Scale factor = 0.0 sets all values to zero
- **Negative scaling**: Handles negative factors correctly
- **Empty dimensions**: Dimensions with no non-zeros are skipped

**Data Guarantees (Preconditions)**

- `scales.len >= matrix.primary_dim()`
- Matrix values must be mutable (non-const)
- Matrix structure must be valid

**MUTABILITY**

INPLACE - modifies `matrix.values()` directly

**Complexity Analysis**

- **Time**: O(nnz) for scaling all values
  - Each non-zero is accessed and scaled once
  - Parallelized over primary dimensions
- **Space**: O(1) auxiliary space

**Example**

```cpp
scl::Sparse<Real, true> matrix = /* ... */;
scl::Array<Real> row_sums(matrix.rows());

// Compute row sums
scl::kernel::normalize::compute_row_sums(matrix, row_sums);

// Compute inverse sums for normalization
scl::Array<Real> inv_sums(matrix.rows());
for (Index i = 0; i < matrix.rows(); ++i) {
    inv_sums[i] = (row_sums[i] > 0) ? 1.0 / row_sums[i] : 0.0;
}

// Scale rows by inverse sums (normalize to sum = 1)
scl::kernel::normalize::scale_primary(matrix, inv_sums);

// Matrix is now row-normalized
```

---

### primary_sums_masked

::: source_code file="scl/kernel/normalize.hpp" symbol="primary_sums_masked" collapsed
:::

**Algorithm Description**

Compute sum of values in each primary dimension, counting only elements where mask is zero:

1. **Mask Checking**: For each non-zero element:
   - Check mask[indices[j]] == 0
   - Only count elements where mask is zero (unmasked)

2. **Conditional Summation**: For each primary dimension:
   - Iterate over non-zero elements
   - Sum only unmasked values
   - Use optimized SIMD path when possible

3. **Output**: Store masked sums in output buffer:
   - `output[i]` = sum of unmasked values in primary dimension i

**Edge Cases**

- **All masked**: Dimensions with all elements masked get sum = 0
- **No mask**: If all mask values are 0, equivalent to compute_row_sums
- **Empty dimensions**: Dimensions with no non-zeros get sum = 0
- **Invalid mask indices**: Mask indices must match secondary dimension

**Data Guarantees (Preconditions)**

- `output.len >= matrix.primary_dim()`
- `mask.len >= matrix.secondary_dim()`
- Matrix must be valid sparse format
- Mask values are 0 (unmasked) or non-zero (masked)

**Complexity Analysis**

- **Time**: O(nnz) for checking mask and summing
  - Each non-zero is checked against mask
  - Parallelized over primary dimensions
- **Space**: O(1) auxiliary space

**Example**

```cpp
scl::Sparse<Real, true> matrix = /* ... */;
scl::Array<Byte> gene_mask(n_genes);  // 0 = include, 1 = exclude

// Set mask to exclude certain genes
for (Index g = 0; g < n_genes; ++g) {
    if (should_exclude(g)) {
        gene_mask[g] = 1;  // Mask this gene
    }
}

scl::Array<Real> masked_row_sums(matrix.rows());

// Compute row sums excluding masked genes
scl::kernel::normalize::primary_sums_masked(
    matrix,
    gene_mask,
    masked_row_sums
);

// masked_row_sums[i] contains sum of row i excluding masked genes
```

---

### detect_highly_expressed

::: source_code file="scl/kernel/normalize.hpp" symbol="detect_highly_expressed" collapsed
:::

**Algorithm Description**

Detect genes that are highly expressed in each cell, where expression exceeds a fraction of total expression:

1. **Threshold Computation**: For each cell:
   - Compute threshold = row_sums[cell] * max_fraction
   - This defines minimum expression level for "highly expressed"

2. **Gene Detection**: For each cell in parallel:
   - Iterate over expressed genes in cell
   - If value > threshold, mark gene as highly expressed
   - Use atomic operations to update mask (thread-safe)

3. **Output**: Store mask in output buffer:
   - `out_mask[g] == 1` if gene g is highly expressed in any cell
   - `out_mask[g] == 0` otherwise

**Edge Cases**

- **Zero row sums**: Cells with row_sum = 0 get threshold = 0, all genes marked
- **Very high fraction**: max_fraction = 1.0 marks all expressed genes
- **Very low fraction**: max_fraction near 0 marks only top-expressed genes
- **Empty cells**: Cells with no expression don't mark any genes

**Data Guarantees (Preconditions)**

- `row_sums.len == matrix.rows()`
- `out_mask.len >= matrix.cols()`
- `max_fraction` in (0, 1]
- Row sums must be pre-computed (e.g., using compute_row_sums)

**Complexity Analysis**

- **Time**: O(nnz) for checking all non-zeros
  - Each non-zero is checked against threshold
  - Atomic operations for mask updates
- **Space**: O(1) auxiliary space

**Example**

```cpp
scl::Sparse<Real, true> expression = /* ... */;  // [n_cells x n_genes]
scl::Array<Real> row_sums(n_cells);

// Compute row sums first
scl::kernel::normalize::compute_row_sums(expression, row_sums);

// Detect genes with > 10% of cell's total expression
Real max_fraction = 0.1;
scl::Array<Byte> highly_expressed(n_genes);

scl::kernel::normalize::detect_highly_expressed(
    expression,
    row_sums,
    max_fraction,
    highly_expressed
);

// highly_expressed[g] == 1 if gene g is highly expressed in any cell
```

---

## Configuration

Default parameters in `scl::kernel::normalize::config`:

- `PREFETCH_DISTANCE = 64`: Cache line prefetch distance for SIMD operations

---

## Performance Notes

### SIMD Optimization

- All operations use SIMD for vectorized computation
- 4-way or 8-way unrolling for long rows/columns
- Prefetching for memory latency hiding

### Parallelization

- All functions parallelize over primary dimensions
- No synchronization needed (distinct output elements)
- Efficient thread-local processing

---

## Use Cases

### Row Normalization

```cpp
// Normalize matrix so each row sums to 1
scl::Array<Real> row_sums(n_rows);
scl::kernel::normalize::compute_row_sums(matrix, row_sums);

scl::Array<Real> inv_sums(n_rows);
for (Index i = 0; i < n_rows; ++i) {
    inv_sums[i] = (row_sums[i] > 0) ? 1.0 / row_sums[i] : 0.0;
}

scl::kernel::normalize::scale_primary(matrix, inv_sums);
```

### Gene Filtering

```cpp
// Detect and filter highly expressed genes
scl::Array<Real> row_sums(n_cells);
scl::kernel::normalize::compute_row_sums(expression, row_sums);

scl::Array<Byte> highly_expressed(n_genes);
scl::kernel::normalize::detect_highly_expressed(
    expression, row_sums, 0.1, highly_expressed
);

// Filter expression matrix to keep only highly expressed genes
```

---

## See Also

- [Sparse Matrices](../core/sparse)
- [SIMD Operations](../core/simd)
- [Memory Management](../core/memory)
