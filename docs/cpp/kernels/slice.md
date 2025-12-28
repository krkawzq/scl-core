# slice.hpp

> scl/kernel/slice.hpp · Sparse matrix slicing kernels

## Overview

This file provides high-performance kernels for slicing sparse matrices along primary and secondary dimensions. It supports efficient inspection (counting non-zeros), materialization (copying to pre-allocated arrays), and full slicing operations that create new sparse matrices. All operations are parallelized and optimized for cache efficiency.

**Header**: `#include "scl/kernel/slice.hpp"`

---

## Main APIs

### slice_primary

::: source_code file="scl/kernel/slice.hpp" symbol="slice_primary" collapsed
:::

**Algorithm Description**

Creates new sparse matrix containing selected primary slices (rows for CSR, columns for CSC):

1. Call `inspect_slice_primary` to count total non-zeros in selected slices
2. Allocate output arrays (data, indices, indptr) with appropriate sizes
3. Call `materialize_slice_primary` to copy selected slices
4. Wrap arrays as new Sparse matrix
5. Result preserves order of `keep_indices` and maintains secondary dimension

**Edge Cases**

- **Empty selection**: Returns empty matrix with zero rows/columns
- **All indices selected**: Returns copy of original matrix
- **Invalid indices**: Behavior undefined if indices out of range
- **Duplicate indices**: Duplicate indices in keep_indices result in duplicate rows/columns

**Data Guarantees (Preconditions)**

- All indices in `keep_indices` are in range [0, primary_dim)
- Source matrix is valid CSR or CSC format
- `keep_indices` may be unsorted

**Complexity Analysis**

- **Time**: O(nnz_output / n_threads + n_keep) - parallel copy plus metadata setup
- **Space**: O(nnz_output) for result matrix

**Example**

```cpp
#include "scl/kernel/slice.hpp"

Sparse<Real, true> matrix = /* source matrix, CSR */;
Array<const Index> keep_indices = /* row indices to keep */;

Sparse<Real, true> sliced = scl::kernel::slice::slice_primary(
    matrix,
    keep_indices
);

// sliced contains only selected rows, columns preserved
```

---

### filter_secondary

::: source_code file="scl/kernel/slice.hpp" symbol="filter_secondary" collapsed
:::

**Algorithm Description**

Creates new sparse matrix filtering by secondary dimension mask (columns for CSR, rows for CSC):

1. Build index mapping from old to new secondary indices (compact range)
2. Call `inspect_filter_secondary` to count non-zeros after filtering
3. Allocate output arrays with appropriate sizes
4. Call `materialize_filter_secondary` to copy and remap indices
5. Result has compact secondary dimension [0, new_secondary_dim)

**Edge Cases**

- **All zeros mask**: Returns empty matrix (zero columns/rows)
- **All ones mask**: Returns copy of original matrix
- **Sparse mask**: Efficiently handles masks with few 1s
- **Index remapping**: Old indices remapped to compact range

**Data Guarantees (Preconditions)**

- `mask.len >= secondary_dim`
- Mask values are 0 or 1
- Source matrix is valid sparse format

**Complexity Analysis**

- **Time**: O(nnz / n_threads + secondary_dim) - parallel filtering plus mapping
- **Space**: O(nnz_output + secondary_dim) for result and index mapping

**Example**

```cpp
Array<const uint8_t> mask = /* boolean mask for columns */;

Sparse<Real, true> filtered = scl::kernel::slice::filter_secondary(
    matrix,
    mask
);

// filtered contains only columns where mask[col] == 1
// Column indices remapped to [0, new_n_cols)
```

---

## Utility Functions

### inspect_slice_primary

Counts total non-zeros in selected primary dimension slices.

::: source_code file="scl/kernel/slice.hpp" symbol="inspect_slice_primary" collapsed
:::

**Complexity**

- Time: O(n_keep / n_threads)
- Space: O(n_threads) for partial sums

---

### materialize_slice_primary

Copies selected primary slices to pre-allocated output arrays.

::: source_code file="scl/kernel/slice.hpp" symbol="materialize_slice_primary" collapsed
:::

**Complexity**

- Time: O(nnz_output / n_threads + n_keep)
- Space: O(1) beyond output

---

### inspect_filter_secondary

Counts non-zeros after filtering by secondary dimension mask.

::: source_code file="scl/kernel/slice.hpp" symbol="inspect_filter_secondary" collapsed
:::

**Complexity**

- Time: O(nnz / n_threads)
- Space: O(n_threads) for partial sums

---

### materialize_filter_secondary

Copies elements passing secondary mask to pre-allocated output with index remapping.

::: source_code file="scl/kernel/slice.hpp" symbol="materialize_filter_secondary" collapsed
:::

**Complexity**

- Time: O(nnz / n_threads + primary_dim)
- Space: O(1) beyond output

---

## Configuration

Internal configuration constants:

- `PARALLEL_THRESHOLD_ROWS = 512`: Minimum rows for parallel processing
- `PARALLEL_THRESHOLD_NNZ = 10000`: Minimum nnz for parallel processing
- `MEMCPY_THRESHOLD = 8`: Minimum elements for memcpy vs loop

---

## Performance Notes

### Parallelization

- Primary dimension slicing: Parallel reduction for counting, parallel copy for materialization
- Secondary dimension filtering: Parallel over primary dimension with 8-way unrolled counting
- Cache-efficient: Uses prefetching and batched processing

### Memory Efficiency

- Two-phase approach: Inspect first to size output, then materialize
- Pre-allocated arrays: Allows caller to manage memory
- Zero-copy potential: Can wrap existing arrays with proper ownership

---

## See Also

- [Sparse Matrix](../core/sparse) - Sparse matrix operations
- [Memory Module](../core/memory) - Memory management
