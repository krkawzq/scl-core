# Sparse Matrix Statistics Kernels

Sparse matrix statistics with SIMD optimization.

**Location**: `scl/kernel/sparse.hpp`

---

## primary_sums

**SUMMARY:**
Compute the sum of values for each primary dimension (row for CSR, column for CSC).

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void primary_sums(
    const Sparse<T, IsCSR>& matrix,    // Input sparse matrix
    Array<T> output                     // Output sums [primary_dim]
);
```

**PARAMETERS:**
- matrix [in]  Sparse matrix, shape (rows, cols)
- output [out] Pre-allocated buffer, size = primary_dim

**PRECONDITIONS:**
- matrix.valid() must be true
- output.len >= matrix.primary_dim()

**POSTCONDITIONS:**
- output[i] contains sum of values in primary slice i
- Empty rows/columns result in zero

**ALGORITHM:**
Uses SIMD-optimized vectorize::sum for each primary dimension slice.

**COMPLEXITY:**
- Time:  O(nnz) with SIMD acceleration
- Space: O(1) auxiliary

**THREAD SAFETY:**
Safe - parallelized over primary dimension

**MUTABILITY:**
CONST - matrix is not modified

---

## primary_means

**SUMMARY:**
Compute the mean of values for each primary dimension (treating implicit zeros).

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void primary_means(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output
);
```

**PARAMETERS:**
- matrix [in]  Sparse matrix, shape (rows, cols)
- output [out] Pre-allocated buffer, size = primary_dim

**PRECONDITIONS:**
- matrix.valid() must be true
- output.len >= matrix.primary_dim()

**POSTCONDITIONS:**
- output[i] contains mean of primary slice i
- Mean is computed over secondary_dim (including implicit zeros)

**ALGORITHM:**
1. Compute sum of non-zero elements
2. Divide by secondary_dim (total elements including zeros)

**COMPLEXITY:**
- Time:  O(nnz)
- Space: O(1) auxiliary

**THREAD SAFETY:**
Safe - parallelized over primary dimension

---

## primary_variances

**SUMMARY:**
Compute variance of values for each primary dimension (treating implicit zeros).

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void primary_variances(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output,
    int ddof = 1                        // Degrees of freedom correction
);
```

**PARAMETERS:**
- matrix [in]  Sparse matrix
- output [out] Variance values [primary_dim]
- ddof   [in]  Delta degrees of freedom (default 1 for sample variance)

**PRECONDITIONS:**
- matrix.valid() must be true
- output.len >= matrix.primary_dim()
- ddof < matrix.secondary_dim()

**POSTCONDITIONS:**
- output[i] contains variance of primary slice i
- Uses denominator (N - ddof) where N = secondary_dim
- Variance is non-negative (clipped at zero for numerical stability)

**ALGORITHM:**
1. Compute sum and sum-of-squares using fused SIMD operations
2. Apply variance formula: var = (sum_sq - sum * mean) / denom
3. Clip negative values (from numerical error) to zero

**COMPLEXITY:**
- Time:  O(nnz) with multi-accumulator SIMD
- Space: O(1) auxiliary

**THREAD SAFETY:**
Safe - parallelized over primary dimension

**NUMERICAL NOTES:**
- Uses compensated summation for numerical stability
- Clamps negative variances to zero

---

## primary_nnz

**SUMMARY:**
Count non-zero elements per primary dimension.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void primary_nnz(
    const Sparse<T, IsCSR>& matrix,
    Array<Index> output
);
```

**PARAMETERS:**
- matrix [in]  Sparse matrix
- output [out] Non-zero counts [primary_dim]

**PRECONDITIONS:**
- matrix.valid() must be true
- output.len >= matrix.primary_dim()

**POSTCONDITIONS:**
- output[i] contains number of non-zero elements in primary slice i

**ALGORITHM:**
Returns primary_length for each dimension (O(1) per element).

**COMPLEXITY:**
- Time:  O(primary_dim)
- Space: O(1) auxiliary

**THREAD SAFETY:**
Safe - parallelized for large matrices

**MUTABILITY:**
CONST - matrix is not modified

---

## to_contiguous_arrays

**SUMMARY:**
Export sparse matrix to traditional CSR/CSC contiguous format.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
ContiguousArraysT<T> to_contiguous_arrays(
    const Sparse<T, IsCSR>& matrix
);
```

**RETURN VALUE:**
```cpp
struct ContiguousArraysT<T> {
    T* data;             // Registry-managed values array
    Index* indices;      // Registry-managed indices array
    Index* indptr;       // Registry-managed offset array
    Index nnz;           // Total non-zeros
    Index primary_dim;   // Number of rows (CSR) or cols (CSC)
};
```

**PARAMETERS:**
- matrix [in] Sparse matrix to export

**PRECONDITIONS:**
- matrix.valid() must be true

**POSTCONDITIONS:**
- Returns contiguous arrays registered with registry
- data, indices, indptr are allocated via registry
- Caller must unregister arrays when done
- For empty matrix, returns nullptr pointers with nnz=0

**ALGORITHM:**
1. Allocate contiguous arrays via registry
2. Build indptr array from lengths
3. Copy data and indices sequentially

**COMPLEXITY:**
- Time:  O(nnz + primary_dim)
- Space: O(nnz + primary_dim)

**THREAD SAFETY:**
Safe - sequential copy operations

**LIFECYCLE:**
ALLOCATES - arrays are registered with registry, caller must unregister

---

## to_coo_arrays

**SUMMARY:**
Export sparse matrix to COO (Coordinate) format.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
COOArraysT<T> to_coo_arrays(
    const Sparse<T, IsCSR>& matrix
);
```

**RETURN VALUE:**
```cpp
struct COOArraysT<T> {
    Index* row_indices;  // Registry-managed row indices [nnz]
    Index* col_indices;  // Registry-managed column indices [nnz]
    T* values;           // Registry-managed values [nnz]
    Index nnz;           // Total non-zeros
};
```

**PARAMETERS:**
- matrix [in] Sparse matrix to export

**PRECONDITIONS:**
- matrix.valid() must be true

**POSTCONDITIONS:**
- Returns COO arrays registered with registry
- For empty matrix, returns nullptr pointers with nnz=0
- Caller must unregister arrays when done

**ALGORITHM:**
1. Allocate COO arrays via registry
2. Parallel conversion: iterate primary dimension, expand coordinates
3. For CSR: row_indices[k] = i, col_indices[k] = indices[k]
4. For CSC: col_indices[k] = j, row_indices[k] = indices[k]

**COMPLEXITY:**
- Time:  O(nnz) parallelized
- Space: O(nnz)

**THREAD SAFETY:**
Safe - parallelized over primary dimension with non-overlapping writes

**LIFECYCLE:**
ALLOCATES - arrays are registered with registry, caller must unregister

---

## from_contiguous_arrays

**SUMMARY:**
Create sparse matrix from traditional CSR/CSC contiguous arrays.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
Sparse<T, IsCSR> from_contiguous_arrays(
    T* data,
    Index* indices,
    Index* indptr,
    Index rows,
    Index cols,
    Index nnz,
    bool take_ownership = false
);
```

**PARAMETERS:**
- data          [in] Values array [nnz]
- indices       [in] Secondary indices array [nnz]
- indptr        [in] Primary offset array [primary_dim+1]
- rows          [in] Number of rows
- cols          [in] Number of columns
- nnz           [in] Total non-zeros
- take_ownership [in] If true, register arrays with registry for automatic cleanup

**PRECONDITIONS:**
- data, indices, indptr must not be null
- indptr[0] == 0
- indptr[primary_dim] == nnz
- indptr is non-decreasing
- indices are sorted within each primary slice

**POSTCONDITIONS:**
- Returns Sparse matrix wrapping the arrays
- If take_ownership=true: arrays are registered with registry as buffers+aliases
- If take_ownership=false: caller manages array lifetime
- Matrix shares memory with input arrays (zero-copy)

**ALGORITHM:**
1. If take_ownership: register buffers and create aliases via registry
2. Build pointer arrays from indptr offsets
3. Wrap using Sparse::wrap_traditional_unsafe

**COMPLEXITY:**
- Time:  O(primary_dim)
- Space: O(primary_dim) for metadata

**THREAD SAFETY:**
Safe - no concurrent modification

**LIFECYCLE:**
- If take_ownership=false: CONST - caller manages data lifetime
- If take_ownership=true: ALLOCATES - registry manages data lifecycle

**MUTABILITY:**
CONST - does not copy data, wraps existing arrays

---

## eliminate_zeros

**SUMMARY:**
Remove elements with absolute value below tolerance.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
Sparse<T, IsCSR> eliminate_zeros(
    const Sparse<T, IsCSR>& matrix,
    T tolerance = T(0)
);
```

**PARAMETERS:**
- matrix    [in] Input sparse matrix
- tolerance [in] Absolute value threshold (default 0)

**PRECONDITIONS:**
- matrix.valid() must be true
- tolerance >= 0

**POSTCONDITIONS:**
- Returns new matrix with elements |x| > tolerance
- Original matrix is unchanged
- Result is in contiguous format

**ALGORITHM:**
1. Parallel count elements above threshold per primary slice
2. Allocate result matrix
3. Parallel copy non-zero elements

**COMPLEXITY:**
- Time:  O(nnz) parallelized
- Space: O(nnz') where nnz' = output nnz

**THREAD SAFETY:**
Safe - parallelized over primary dimension

**MUTABILITY:**
CONST - creates new matrix, does not modify input

---

## prune

**SUMMARY:**
Remove or zero elements with absolute value below threshold.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
Sparse<T, IsCSR> prune(
    const Sparse<T, IsCSR>& matrix,
    T threshold,
    bool keep_structure = false
);
```

**PARAMETERS:**
- matrix         [in] Input sparse matrix
- threshold      [in] Absolute value cutoff
- keep_structure [in] If true, set small values to zero but keep structure

**PRECONDITIONS:**
- matrix.valid() must be true
- threshold >= 0

**POSTCONDITIONS:**
- If keep_structure=false: removes elements |x| < threshold (compact)
- If keep_structure=true: sets elements |x| < threshold to explicit zeros
- Original matrix is unchanged

**ALGORITHM:**
- If keep_structure=false: calls eliminate_zeros(matrix, threshold)
- If keep_structure=true: clone matrix and zero small elements in-place

**COMPLEXITY:**
- Time:  O(nnz)
- Space: O(nnz)

**THREAD SAFETY:**
Safe - uses safe operations

**MUTABILITY:**
CONST - creates new matrix

---

## validate

**SUMMARY:**
Validate sparse matrix format and data integrity.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
ValidationResult validate(
    const Sparse<T, IsCSR>& matrix
);
```

**RETURN VALUE:**
```cpp
struct ValidationResult {
    bool valid;
    const char* error_message;  // nullptr if valid
    Index error_index;          // -1 if valid, else primary index with error
};
```

**PARAMETERS:**
- matrix [in] Sparse matrix to validate

**PRECONDITIONS:**
None

**POSTCONDITIONS:**
- Returns validation result
- If invalid, provides error message and index

**ALGORITHM:**
1. Check matrix.valid()
2. Verify all indices are in range [0, secondary_dim)
3. Verify indices are strictly sorted within each primary slice
4. Verify total nnz matches sum of lengths

**COMPLEXITY:**
- Time:  O(nnz + primary_dim)
- Space: O(1) auxiliary

**THREAD SAFETY:**
Safe - read-only

**MUTABILITY:**
CONST - does not modify matrix

---

## memory_info

**SUMMARY:**
Get memory usage information for sparse matrix.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
MemoryInfo memory_info(
    const Sparse<T, IsCSR>& matrix
);
```

**RETURN VALUE:**
```cpp
struct MemoryInfo {
    Size data_bytes;        // Bytes for values
    Size indices_bytes;     // Bytes for indices
    Size metadata_bytes;    // Bytes for pointers and lengths
    Size total_bytes;       // Sum of above
    Index block_count;      // Number of data blocks
    bool is_contiguous;     // True if single contiguous block
};
```

**PARAMETERS:**
- matrix [in] Sparse matrix to analyze

**PRECONDITIONS:**
None

**POSTCONDITIONS:**
- Returns memory usage statistics
- For invalid matrix, returns zeros

**COMPLEXITY:**
- Time:  O(primary_dim) for block counting
- Space: O(1)

**THREAD SAFETY:**
Safe - read-only

**MUTABILITY:**
CONST

---

## make_contiguous

**SUMMARY:**
Convert sparse matrix to contiguous storage if not already.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
Sparse<T, IsCSR> make_contiguous(
    const Sparse<T, IsCSR>& matrix
);
```

**PARAMETERS:**
- matrix [in] Input sparse matrix

**PRECONDITIONS:**
- matrix.valid() must be true

**POSTCONDITIONS:**
- Returns matrix with contiguous storage
- If already contiguous, returns clone
- Original matrix unchanged

**ALGORITHM:**
Calls matrix.to_contiguous() or matrix.clone(BlockStrategy::contiguous())

**COMPLEXITY:**
- Time:  O(nnz + primary_dim)
- Space: O(nnz + primary_dim)

**THREAD SAFETY:**
Safe

**MUTABILITY:**
CONST - creates new matrix

---

## resize_secondary

**SUMMARY:**
Change secondary dimension size (columns for CSR, rows for CSC).

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void resize_secondary(
    Sparse<T, IsCSR>& matrix,
    Index new_secondary_dim
);
```

**PARAMETERS:**
- matrix            [in,out] Sparse matrix to modify
- new_secondary_dim [in]     New secondary dimension size

**PRECONDITIONS:**
- matrix.valid() must be true
- new_secondary_dim >= 0
- If shrinking, all indices must be < new_secondary_dim

**POSTCONDITIONS:**
- Matrix secondary dimension is updated
- Data and indices are unchanged
- If shrinking, caller must ensure no indices are out of bounds

**ALGORITHM:**
Updates dimension metadata only (cols_ for CSR, rows_ for CSC).

**COMPLEXITY:**
- Time:  O(1) in release, O(nnz) validation in debug
- Space: O(1)

**THREAD SAFETY:**
Unsafe - modifies matrix in-place

**MUTABILITY:**
INPLACE - modifies dimension metadata

**NUMERICAL NOTES:**
In debug mode, validates no indices are out of bounds when shrinking
