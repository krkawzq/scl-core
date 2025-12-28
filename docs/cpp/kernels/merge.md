# Matrix Merging

Vertically and horizontally stack sparse matrices.

## Overview

Merge operations provide:

- **Vertical stacking** - Concatenate matrices along primary axis (rows for CSR, columns for CSC)
- **Horizontal stacking** - Concatenate matrices along secondary axis (columns for CSR, rows for CSC)
- **Efficient copying** - Parallel memory operations with SIMD optimization
- **Memory management** - Flexible block allocation strategies

## Vertical Stacking

### vstack

Vertically stack two sparse matrices (concatenate along primary axis).

```cpp
#include "scl/kernel/merge.hpp"
#include "scl/core/sparse.hpp"

Sparse<Real, true> matrix1 = /* ... */;  // First matrix
Sparse<Real, true> matrix2 = /* ... */;  // Second matrix

auto result = scl::kernel::merge::vstack(matrix1, matrix2);
// result contains vertically stacked matrix
```

**Parameters:**
- `matrix1` [in] - First sparse matrix
- `matrix2` [in] - Second sparse matrix
- `strategy` [in] - Block allocation strategy for result (default: adaptive)

**Preconditions:**
- For CSR: columns can differ (result uses max)
- For CSC: rows can differ (result uses max)

**Postconditions:**
- Result primary_dim = matrix1.primary_dim + matrix2.primary_dim
- Result secondary_dim = max(matrix1.secondary_dim, matrix2.secondary_dim)
- Rows 0..n1-1 from matrix1, rows n1..n1+n2-1 from matrix2
- Indices unchanged (secondary dimension preserved)

**Returns:**
New sparse matrix with vertically stacked data

**Algorithm:**
1. Compute row lengths for result
2. Allocate result matrix with combined structure
3. Parallel copy matrix1 rows to result[0:n1]
4. Parallel copy matrix2 rows to result[n1:n1+n2]

**Complexity:**
- Time: O(nnz1 + nnz2)
- Space: O(nnz1 + nnz2) for result

**Thread Safety:**
Safe - parallel copy of independent regions

**Use cases:**
- Combining datasets with same features
- Appending new samples to existing matrix
- Merging time series data

## Horizontal Stacking

### hstack

Horizontally stack two sparse matrices (concatenate along secondary axis).

```cpp
auto result = scl::kernel::merge::hstack(matrix1, matrix2);
// result contains horizontally stacked matrix
```

**Parameters:**
- `matrix1` [in] - First sparse matrix
- `matrix2` [in] - Second sparse matrix
- `strategy` [in] - Block allocation strategy for result (default: adaptive)

**Preconditions:**
- matrix1.primary_dim == matrix2.primary_dim (must match)

**Postconditions:**
- Result primary_dim = matrix1.primary_dim (unchanged)
- Result secondary_dim = matrix1.secondary_dim + matrix2.secondary_dim
- For each row: [matrix1 columns | matrix2 columns with offset]
- matrix2 indices offset by matrix1.secondary_dim

**Returns:**
New sparse matrix with horizontally stacked data

**Algorithm:**
1. Verify primary dimensions match
2. Compute combined row lengths
3. Allocate result matrix
4. Parallel over rows:
   - Copy matrix1 values and indices
   - Copy matrix2 values
   - Add offset to matrix2 indices (SIMD optimized)

**Complexity:**
- Time: O(nnz1 + nnz2)
- Space: O(nnz1 + nnz2) for result

**Thread Safety:**
Safe - parallel over independent rows

**Throws:**
`DimensionError` - if primary dimensions mismatch

**Use cases:**
- Combining feature sets
- Concatenating gene expression from different batches
- Merging matrices with same samples

## Examples

### Combining Datasets

Combine two expression matrices with same genes:

```cpp
Sparse<Real, true> batch1 = /* ... */;  // cells x genes
Sparse<Real, true> batch2 = /* ... */;  // cells x genes

// Vertically stack (combine cells)
auto combined = scl::kernel::merge::vstack(batch1, batch2);
// combined has (batch1.rows() + batch2.rows()) rows
```

### Concatenating Features

Combine matrices with same cells but different features:

```cpp
Sparse<Real, true> rna = /* ... */;    // cells x RNA genes
Sparse<Real, true> protein = /* ... */; // cells x proteins

// Horizontally stack (combine features)
auto multiome = scl::kernel::merge::hstack(rna, protein);
// multiome has rna.rows() rows and (rna.cols() + protein.cols()) columns
```

### Memory Strategy

Choose allocation strategy for large matrices:

```cpp
// Use adaptive strategy (default)
auto result1 = scl::kernel::merge::vstack(m1, m2);

// Use specific strategy
auto result2 = scl::kernel::merge::vstack(
    m1, m2,
    BlockStrategy::contiguous()  // Force contiguous allocation
);
```

## Performance

### Parallelization

- Parallel memory copy for large blocks
- Independent row processing
- No synchronization overhead

### SIMD Optimization

- SIMD-optimized index offset addition
- Prefetch in copy loops
- Efficient memory access patterns

### Memory Efficiency

- Adaptive block allocation
- Minimal intermediate allocations
- Efficient sparse matrix construction

## Implementation Details

### Index Offset Addition

For horizontal stacking, matrix2 indices are offset by matrix1.secondary_dim using SIMD-optimized addition:

```cpp
// offset == 0: direct memcpy (early exit)
// Otherwise: 2-way SIMD unrolled loop
```

### Parallel Memory Copy

Large memory blocks use parallel copying:
- count < chunk_size: single memcpy
- Otherwise: parallel_for over chunks with prefetch
