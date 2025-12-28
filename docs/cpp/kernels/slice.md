# Matrix Slicing

Slice and filter sparse matrices along primary or secondary dimensions.

## Overview

Slice operations provide:

- **Primary dimension slicing** - Select specific rows (CSR) or columns (CSC)
- **Secondary dimension filtering** - Filter by columns (CSR) or rows (CSC) using boolean mask
- **Efficient inspection** - Count non-zeros before allocation
- **Memory efficient** - Two-phase approach (inspect then materialize)

## Primary Dimension Slicing

### slice_primary

Create new sparse matrix containing selected primary slices.

```cpp
#include "scl/kernel/slice.hpp"
#include "scl/core/sparse.hpp"

Sparse<Real, true> matrix = /* ... */;
Array<Index> keep_indices = /* ... */;  // Indices of rows to keep

auto result = scl::kernel::slice::slice_primary(matrix, keep_indices);
// result contains only selected rows
```

**Parameters:**
- `matrix` [in] - Source sparse matrix
- `keep_indices` [in] - Indices of rows (CSR) or cols (CSC) to keep

**Preconditions:**
- All indices in range [0, primary_dim)

**Postconditions:**
- Result contains only selected rows/cols
- Column/row indices unchanged (secondary dim preserved)
- Order matches keep_indices order

**Returns:**
New sparse matrix with selected slices

**Algorithm:**
1. inspect_slice_primary to count output nnz
2. Allocate output arrays
3. materialize_slice_primary to copy data
4. Wrap as new Sparse matrix

**Complexity:**
- Time: O(nnz_output / n_threads + n_keep)
- Space: O(nnz_output) for result

**Thread Safety:**
Safe - uses parallel materialize

**Use cases:**
- Selecting subset of samples/cells
- Filtering by metadata
- Creating training/test splits

### inspect_slice_primary

Count total non-zeros in selected primary dimension slices.

```cpp
Index nnz_output = scl::kernel::slice::inspect_slice_primary(
    matrix,
    keep_indices
);
// Returns total number of non-zeros in selected slices
```

**Parameters:**
- `matrix` [in] - Sparse matrix to slice
- `keep_indices` [in] - Indices of primary dimension elements to keep

**Preconditions:**
- All indices in keep_indices in range [0, primary_dim)

**Postconditions:**
- Returns sum of row lengths for selected indices

**Returns:**
Total number of non-zeros in selected slices

**Algorithm:**
Parallel reduction over keep_indices using parallel_reduce_nnz

**Complexity:**
- Time: O(n_keep / n_threads)
- Space: O(n_threads) for partial sums

**Thread Safety:**
Safe - read-only parallel reduction

**Use cases:**
- Pre-allocating output arrays
- Estimating memory requirements

### materialize_slice_primary

Copy selected primary slices to pre-allocated output arrays.

```cpp
Array<Real> out_data(nnz_output);
Array<Index> out_indices(nnz_output);
Array<Index> out_indptr(keep_indices.len + 1);

scl::kernel::slice::materialize_slice_primary(
    matrix,
    keep_indices,
    out_data,
    out_indices,
    out_indptr
);
```

**Parameters:**
- `matrix` [in] - Source sparse matrix
- `keep_indices` [in] - Indices of rows/cols to keep
- `out_data` [out] - Output values array
- `out_indices` [out] - Output column/row indices array
- `out_indptr` [out] - Output row/col pointer array

**Preconditions:**
- out_data.len >= inspect_slice_primary result
- out_indices.len >= inspect_slice_primary result
- out_indptr.len >= keep_indices.len + 1

**Postconditions:**
- out_data contains copied values in order
- out_indices contains copied indices (unchanged)
- out_indptr[i] = start of i-th selected row

**Algorithm:**
1. Sequential scan to build out_indptr
2. Parallel copy of data and indices using fast_copy_with_prefetch

**Complexity:**
- Time: O(nnz_output / n_threads + n_keep)
- Space: O(1) beyond output

**Thread Safety:**
Safe - parallel copy to disjoint output regions

## Secondary Dimension Filtering

### filter_secondary

Create new sparse matrix filtering by secondary dimension mask.

```cpp
Array<uint8_t> mask(secondary_dim);  // 1 = keep, 0 = remove
// ... fill mask ...

auto result = scl::kernel::slice::filter_secondary(matrix, mask);
// result contains only elements where mask[index] == 1
```

**Parameters:**
- `matrix` [in] - Source sparse matrix
- `mask` [in] - Boolean mask for columns (CSR) or rows (CSC)

**Preconditions:**
- mask.len >= secondary_dim
- mask values are 0 or 1

**Postconditions:**
- Result secondary_dim = count of 1s in mask
- Only elements with mask[index] == 1 retained
- Indices remapped to compact range [0, new_secondary_dim)

**Returns:**
New sparse matrix with filtered secondary dimension

**Algorithm:**
1. Build index mapping (old -> new indices)
2. inspect_filter_secondary to count output nnz
3. Allocate output arrays
4. materialize_filter_secondary to copy and remap

**Complexity:**
- Time: O(nnz / n_threads + secondary_dim)
- Space: O(nnz_output + secondary_dim)

**Thread Safety:**
Safe - uses parallel materialize

**Use cases:**
- Selecting subset of features/genes
- Filtering by expression threshold
- Feature selection

### inspect_filter_secondary

Count non-zeros after filtering by secondary dimension mask.

```cpp
Index nnz_output = scl::kernel::slice::inspect_filter_secondary(
    matrix,
    mask
);
// Returns count of elements where mask[index] == 1
```

**Parameters:**
- `matrix` [in] - Sparse matrix to filter
- `mask` [in] - Boolean mask for secondary dimension (1 = keep)

**Preconditions:**
- mask.len >= secondary_dim
- mask values are 0 or 1

**Postconditions:**
- Returns count of elements where mask[index] == 1

**Returns:**
Total non-zeros after filtering

**Algorithm:**
Parallel reduction using count_masked_fast (8-way unrolled)

**Complexity:**
- Time: O(nnz / n_threads)
- Space: O(n_threads) for partial sums

**Thread Safety:**
Safe - read-only parallel reduction

### materialize_filter_secondary

Copy elements passing secondary mask to pre-allocated output.

```cpp
Array<Index> new_indices = /* build from mask */;
Array<Real> out_data(nnz_output);
Array<Index> out_indices(nnz_output);
Array<Index> out_indptr(primary_dim + 1);

scl::kernel::slice::materialize_filter_secondary(
    matrix,
    mask,
    new_indices,
    out_data,
    out_indices,
    out_indptr
);
```

**Parameters:**
- `matrix` [in] - Source sparse matrix
- `mask` [in] - Boolean mask for secondary dimension
- `new_indices` [in] - Mapping from old to new secondary indices
- `out_data` [out] - Output values
- `out_indices` [out] - Output indices (remapped)
- `out_indptr` [out] - Output row pointers

**Preconditions:**
- new_indices built via build_index_mapping
- Output arrays sized per inspect_filter_secondary

**Postconditions:**
- out_data contains values where mask[old_index] == 1
- out_indices contains remapped indices via new_indices
- out_indptr contains cumulative counts

**Complexity:**
- Time: O(nnz / n_threads + primary_dim)
- Space: O(1) beyond output

**Thread Safety:**
Safe - parallel over primary dimension

## Examples

### Selecting Cells

Select specific cells by index:

```cpp
Sparse<Real, true> expression = /* ... */;  // cells x genes
Array<Index> selected_cells = {0, 5, 10, 15, /* ... */};

auto subset = scl::kernel::slice::slice_primary(expression, selected_cells);
// subset contains only selected cells
```

### Filtering Genes

Filter genes by expression threshold:

```cpp
Sparse<Real, true> expression = /* ... */;  // cells x genes
Array<uint8_t> gene_mask(expression.cols());

// Build mask: keep genes with mean expression > threshold
for (Index g = 0; g < expression.cols(); ++g) {
    Real mean_expr = /* compute mean */;
    gene_mask[g] = (mean_expr > threshold) ? 1 : 0;
}

auto filtered = scl::kernel::slice::filter_secondary(expression, gene_mask);
// filtered contains only high-expression genes
```

### Two-Phase Approach

Use inspect then materialize for memory efficiency:

```cpp
// Phase 1: Count non-zeros
Index nnz = scl::kernel::slice::inspect_slice_primary(matrix, keep_indices);

// Phase 2: Allocate and copy
Array<Real> out_data(nnz);
Array<Index> out_indices(nnz);
Array<Index> out_indptr(keep_indices.len + 1);

scl::kernel::slice::materialize_slice_primary(
    matrix, keep_indices, out_data, out_indices, out_indptr
);
```

## Performance

### Parallelization

- Parallel reduction for inspection
- Parallel copy for materialization
- No synchronization overhead

### SIMD Optimization

- 8-way unrolled mask counting
- Prefetch in copy loops
- Efficient memory access patterns

### Memory Efficiency

- Two-phase approach reduces memory usage
- Pre-allocate output arrays
- Minimal intermediate allocations

## Implementation Details

### Mask Counting

Uses 8-way scalar unroll for counting masked elements:
- Indirect access mask[indices[k]] prevents SIMD gather
- 8-way scalar unroll provides best ILP for this pattern

### Index Mapping

Builds old-to-new index mapping from boolean mask:
- new_indices[i] = new compact index if mask[i] == 1
- new_indices[i] = -1 if mask[i] == 0
- Returns count of 1s in mask
