# Sparse Matrix Slicing Kernels

Optimized parallel sparse matrix slicing operations.

**Location**: `scl/kernel/slice.hpp`

---

## inspect_slice_primary

**SUMMARY:**
Count total non-zeros in primary dimension slice without materializing.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
Index inspect_slice_primary(
    const Sparse<T, IsCSR>& matrix,     // Input matrix
    Array<const Index> keep_indices     // Indices to keep
);
```

**PARAMETERS:**
- matrix       [in] Source sparse matrix
- keep_indices [in] Primary indices to keep (rows for CSR, cols for CSC)

**PRECONDITIONS:**
- matrix.valid() must be true
- All keep_indices[i] in [0, matrix.primary_dim())

**POSTCONDITIONS:**
- Returns total nnz in sliced result

**ALGORITHM:**
Parallel reduction: sum primary_length over keep_indices.

**COMPLEXITY:**
- Time:  O(keep_indices.len) parallelized
- Space: O(num_threads) for reduction

**THREAD SAFETY:**
Safe - read-only with parallel reduction

**MUTABILITY:**
CONST - does not modify input

---

## materialize_slice_primary

**SUMMARY:**
Copy primary dimension slice into pre-allocated contiguous arrays.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void materialize_slice_primary(
    const Sparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices,
    Array<T> out_data,                   // Pre-allocated [out_nnz]
    Array<Index> out_indices,            // Pre-allocated [out_nnz]
    Array<Index> out_indptr              // Pre-allocated [n_keep+1]
);
```

**PARAMETERS:**
- matrix       [in]  Source matrix
- keep_indices [in]  Indices to keep
- out_data     [out] Values array (pre-allocated)
- out_indices  [out] Secondary indices (pre-allocated)
- out_indptr   [out] Offset array (pre-allocated)

**PRECONDITIONS:**
- matrix.valid() must be true
- All keep_indices[i] in valid range
- out_data.len >= total nnz in slice
- out_indices.len >= total nnz in slice
- out_indptr.len >= keep_indices.len + 1

**POSTCONDITIONS:**
- out_indptr[0] = 0
- out_indptr[i+1] = out_indptr[i] + length of primary slice keep_indices[i]
- out_data and out_indices contain copied slice data

**ALGORITHM:**
1. Build offset array from lengths
2. Parallel copy of data and indices slices

**COMPLEXITY:**
- Time:  O(keep_indices.len + out_nnz) parallelized
- Space: O(1)

**THREAD SAFETY:**
Safe - non-overlapping writes in parallel

**MUTABILITY:**
CONST for matrix, writes to output arrays

---

## slice_primary

**SUMMARY:**
Extract primary dimension slice as new sparse matrix.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
Sparse<T, IsCSR> slice_primary(
    const Sparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices
);
```

**PARAMETERS:**
- matrix       [in] Source matrix
- keep_indices [in] Primary indices to keep (rows for CSR, cols for CSC)

**PRECONDITIONS:**
- matrix.valid() must be true
- All keep_indices[i] in [0, matrix.primary_dim())

**POSTCONDITIONS:**
- Returns new matrix with selected rows/columns
- Result dimensions: (n_keep, cols) for CSR, (rows, n_keep) for CSC
- Result is in contiguous format
- Memory registered with registry for automatic cleanup

**ALGORITHM:**
1. Inspect to compute output nnz
2. Allocate output arrays via aligned_alloc
3. Materialize slice data
4. Register arrays with registry
5. Wrap as Sparse matrix

**COMPLEXITY:**
- Time:  O(n_keep + out_nnz) parallelized
- Space: O(out_nnz + n_keep)

**THREAD SAFETY:**
Safe - parallel with non-overlapping writes

**MUTABILITY:**
CONST - creates new matrix

**LIFECYCLE:**
ALLOCATES - returns matrix with registry-managed memory

---

## inspect_filter_secondary

**SUMMARY:**
Count total non-zeros in secondary dimension filter without materializing.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
Index inspect_filter_secondary(
    const Sparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask            // Mask for secondary indices
);
```

**PARAMETERS:**
- matrix [in] Source matrix
- mask   [in] Binary mask [secondary_dim], 1=keep, 0=discard

**PRECONDITIONS:**
- matrix.valid() must be true
- mask.len >= matrix.secondary_dim()

**POSTCONDITIONS:**
- Returns total nnz in filtered result
- Count includes only elements with mask[indices[k]] == 1

**ALGORITHM:**
Parallel reduction: for each primary slice, count masked elements.

**COMPLEXITY:**
- Time:  O(nnz) parallelized with unrolled masked counting
- Space: O(num_threads)

**THREAD SAFETY:**
Safe - parallel read-only reduction

**MUTABILITY:**
CONST

---

## materialize_filter_secondary

**SUMMARY:**
Copy secondary dimension filter into pre-allocated arrays with index remapping.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void materialize_filter_secondary(
    const Sparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask,
    Array<const Index> new_indices,      // Remapped indices [secondary_dim]
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr
);
```

**PARAMETERS:**
- matrix      [in]  Source matrix
- mask        [in]  Binary mask [secondary_dim]
- new_indices [in]  Remapped index array: new_indices[old] = new or -1
- out_data    [out] Values array (pre-allocated)
- out_indices [out] Remapped secondary indices (pre-allocated)
- out_indptr  [out] Offset array (pre-allocated)

**PRECONDITIONS:**
- matrix.valid() must be true
- mask.len >= matrix.secondary_dim()
- new_indices.len >= matrix.secondary_dim()
- Output arrays have sufficient size

**POSTCONDITIONS:**
- out_indptr built from masked counts
- out_data and out_indices contain filtered elements
- out_indices remapped via new_indices array

**ALGORITHM:**
1. Build offset array by counting masked elements per primary slice
2. Parallel copy: for each primary slice, copy masked elements with index remapping

**COMPLEXITY:**
- Time:  O(nnz) parallelized
- Space: O(1)

**THREAD SAFETY:**
Safe - non-overlapping writes

**MUTABILITY:**
CONST for matrix, writes to output arrays

---

## filter_secondary

**SUMMARY:**
Filter secondary dimension (columns for CSR, rows for CSC) by mask.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
Sparse<T, IsCSR> filter_secondary(
    const Sparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask
);
```

**PARAMETERS:**
- matrix [in] Source matrix
- mask   [in] Binary mask [secondary_dim], 1=keep, 0=discard

**PRECONDITIONS:**
- matrix.valid() must be true
- mask.len >= matrix.secondary_dim()

**POSTCONDITIONS:**
- Returns new matrix with filtered secondary dimension
- Result dimensions: (rows, new_cols) for CSR, (new_rows, cols) for CSC
- where new_cols/new_rows = number of 1s in mask
- Result is in contiguous format
- Memory registered with registry

**ALGORITHM:**
1. Build index remapping: new_indices[old] = new_idx or -1
2. Compute new_secondary_dim = count(mask == 1)
3. Inspect to compute output nnz
4. Allocate output arrays
5. Materialize filtered data with index remapping
6. Register arrays with registry
7. Wrap as Sparse matrix

**COMPLEXITY:**
- Time:  O(secondary_dim + nnz) parallelized
- Space: O(secondary_dim + out_nnz)

**THREAD SAFETY:**
Safe

**MUTABILITY:**
CONST - creates new matrix

**LIFECYCLE:**
ALLOCATES - returns matrix with registry-managed memory

**NUMERICAL NOTES:**
- Efficient 8-way unrolled masked counting
- Uses prefetching for large data copies
- Automatic parallelization threshold tuning

---

## Configuration Constants

**PARALLEL_THRESHOLD_ROWS:**
Minimum number of primary slices for parallel execution (512).

**PARALLEL_THRESHOLD_NNZ:**
Minimum nnz for parallel operations (10000).

**MEMCPY_THRESHOLD:**
Minimum element count for std::memcpy vs scalar loop (8).

---

## Implementation Details

**Fast Copy with Prefetch:**
- Uses SCL_PREFETCH_READ for sequential access
- Switches between memcpy (large) and scalar loop (small)
- Prefetch distance: 16 cache lines ahead

**Parallel Bulk Copy:**
- Segments work by primary dimension
- Each thread handles independent memory regions
- Automatic threshold-based sequential fallback

**Masked Counting:**
- 8-way unrolled loop for branch-free counting
- Exploits ILP (instruction-level parallelism)
- Scalar cleanup for remainder

**Index Remapping:**
- Sequential scan to build mapping array
- O(secondary_dim) time, cache-friendly
- Remapping done during copy (no extra pass)

**Parallel Reduction:**
- Per-thread partial sums
- Cache-line aligned accumulator array
- Final reduction over thread results

---

## Usage Patterns

**Primary Slicing (Fast Path):**
For row slicing (CSR) or column slicing (CSC), use slice_primary.
This is a zero-copy operation with efficient parallel copy.

**Secondary Filtering (Slower Path):**
For column filtering (CSR) or row filtering (CSC), use filter_secondary.
This requires index remapping but is still parallelized efficiently.

**Two-Phase Inspection:**
1. Call inspect_* to compute output size
2. Allocate output buffers
3. Call materialize_* to fill buffers
This pattern is useful for custom memory management.

**Registry Integration:**
All slice_* and filter_* functions return matrices with registry-managed memory.
Arrays are registered using from_contiguous_arrays with take_ownership=true.
