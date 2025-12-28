// =============================================================================
// FILE: scl/kernel/sparse.h
// BRIEF: API reference for sparse matrix statistics kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::sparse {

/* -----------------------------------------------------------------------------
 * FUNCTION: primary_sums
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute the sum of values along each primary dimension (row for CSR,
 *     column for CSC).
 *
 * PARAMETERS:
 *     matrix [in]  Sparse matrix in CSR or CSC format
 *     output [out] Pre-allocated buffer for sums, size = primary_dim
 *
 * PRECONDITIONS:
 *     - output.len == matrix.primary_dim()
 *     - matrix is valid sparse format
 *
 * POSTCONDITIONS:
 *     - output[i] = sum of all non-zero values in primary slice i
 *     - Empty slices have output[i] = 0
 *     - Matrix unchanged
 *
 * ALGORITHM:
 *     For each primary index in parallel:
 *         1. Get values span for the primary slice
 *         2. Use scl::vectorize::sum for SIMD-optimized reduction
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimension, no shared mutable state
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void primary_sums(
    const Sparse<T, IsCSR>& matrix,   // Input sparse matrix
    Array<T> output                    // Output sums [primary_dim]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: primary_means
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute the mean of values along each primary dimension, accounting
 *     for implicit zeros (dividing by secondary_dim, not nnz).
 *
 * PARAMETERS:
 *     matrix [in]  Sparse matrix
 *     output [out] Pre-allocated buffer for means
 *
 * PRECONDITIONS:
 *     - output.len == matrix.primary_dim()
 *     - matrix.secondary_dim() > 0
 *
 * POSTCONDITIONS:
 *     - output[i] = sum(primary_slice_i) / secondary_dim
 *     - Empty slices have output[i] = 0
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimension
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void primary_means(
    const Sparse<T, IsCSR>& matrix,   // Input sparse matrix
    Array<T> output                    // Output means [primary_dim]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: primary_variances
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute variance along each primary dimension using fused sum and
 *     sum-of-squares computation.
 *
 * PARAMETERS:
 *     matrix [in]  Sparse matrix
 *     output [out] Pre-allocated buffer for variances
 *     ddof   [in]  Delta degrees of freedom (default 1 for sample variance)
 *
 * PRECONDITIONS:
 *     - output.len == matrix.primary_dim()
 *     - ddof >= 0 and ddof < secondary_dim
 *
 * POSTCONDITIONS:
 *     - output[i] = var(primary_slice_i) with ddof adjustment
 *     - Variance clamped to >= 0 for numerical stability
 *
 * ALGORITHM:
 *     For each primary index in parallel:
 *         1. Use SIMD fused sum+sumsq helper (4-way unroll with FMA)
 *         2. Compute variance = (sumsq - sum*mean) / (N - ddof)
 *         3. Clamp negative values to zero
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimension
 *
 * NUMERICAL NOTES:
 *     Uses compensated summation pattern for improved accuracy
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void primary_variances(
    const Sparse<T, IsCSR>& matrix,   // Input sparse matrix
    Array<T> output,                   // Output variances [primary_dim]
    int ddof = 1                       // Degrees of freedom adjustment
);

/* -----------------------------------------------------------------------------
 * FUNCTION: primary_nnz
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Get the number of non-zero elements in each primary slice.
 *
 * PARAMETERS:
 *     matrix [in]  Sparse matrix
 *     output [out] Pre-allocated buffer for nnz counts
 *
 * PRECONDITIONS:
 *     - output.len == matrix.primary_dim()
 *
 * POSTCONDITIONS:
 *     - output[i] = number of stored elements in primary slice i
 *
 * ALGORITHM:
 *     - For small matrices (< PARALLEL_THRESHOLD): sequential loop
 *     - For large matrices: parallel batched processing (BATCH_SIZE rows per task)
 *
 * COMPLEXITY:
 *     Time:  O(primary_dim)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with batching for cache efficiency
 *
 * PERFORMANCE NOTES:
 *     Uses batched parallel processing to reduce scheduling overhead
 *     for this lightweight operation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void primary_nnz(
    const Sparse<T, IsCSR>& matrix,   // Input sparse matrix
    Array<Index> output                // Output nnz counts [primary_dim]
);

// =============================================================================
// Format Export Tools
// =============================================================================

/* -----------------------------------------------------------------------------
 * STRUCT: ContiguousArraysT<T>
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Generic result structure for to_contiguous_arrays containing
 *     registry-registered pointers to CSR/CSC format arrays.
 *
 * TEMPLATE PARAMETERS:
 *     T - Element type (typically Real, float, or double)
 *
 * MEMBERS:
 *     data      - Pointer to values array (registry registered)
 *     indices   - Pointer to indices array (registry registered)
 *     indptr    - Pointer to offset array (registry registered)
 *     nnz       - Total number of non-zeros
 *     primary_dim - Primary dimension size
 *
 * OWNERSHIP:
 *     All pointers are registered with HandlerRegistry and must be unregistered
 *     when transferring ownership to Python or other external code.
 * -------------------------------------------------------------------------- */
template <typename T>
struct ContiguousArraysT {
    T* data;             // registry registered values array
    Index* indices;      // registry registered indices array
    Index* indptr;       // registry registered offset array
    Index nnz;
    Index primary_dim;
};

/* -----------------------------------------------------------------------------
 * STRUCT: COOArraysT<T>
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Generic result structure for to_coo_arrays containing registry-registered
 *     pointers to COO format arrays.
 *
 * TEMPLATE PARAMETERS:
 *     T - Element type (typically Real, float, or double)
 *
 * MEMBERS:
 *     row_indices - Pointer to row indices array (registry registered)
 *     col_indices - Pointer to column indices array (registry registered)
 *     values      - Pointer to values array (registry registered)
 *     nnz         - Total number of non-zeros
 *
 * OWNERSHIP:
 *     All pointers are registered with HandlerRegistry.
 * -------------------------------------------------------------------------- */
template <typename T>
struct COOArraysT {
    Index* row_indices;  // registry registered
    Index* col_indices;  // registry registered
    T* values;           // registry registered
    Index nnz;
};

/* -----------------------------------------------------------------------------
 * TYPE ALIASES
 * -----------------------------------------------------------------------------
 * ContiguousArrays - Alias for ContiguousArraysT<Real>
 * COOArrays        - Alias for COOArraysT<Real>
 * -------------------------------------------------------------------------- */
using ContiguousArrays = ContiguousArraysT<Real>;
using COOArrays = COOArraysT<Real>;

/* -----------------------------------------------------------------------------
 * FUNCTION: to_contiguous_arrays
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Export sparse matrix to contiguous CSR/CSC format with registry-registered
 *     arrays. Returns a triple of (data, indices, indptr) suitable for Python
 *     bindings.
 *
 * PARAMETERS:
 *     matrix [in] Sparse matrix to export
 *
 * PRECONDITIONS:
 *     - matrix is valid
 *
 * POSTCONDITIONS:
 *     - Returns ContiguousArraysT<T> with registry-registered pointers
 *     - data[i] contains value at position i in row-major order
 *     - indices[i] contains column (CSR) or row (CSC) index
 *     - indptr[i] contains offset for primary dimension i
 *     - indptr[primary_dim] == nnz
 *     - If nnz == 0: data and indices are nullptr, indptr is allocated
 *     - Matrix unchanged
 *
 * RETURNS:
 *     ContiguousArraysT<T> structure with registered pointers, or all-null on failure
 *
 * MUTABILITY:
 *     ALLOCATES - creates new registry-registered arrays
 *
 * ALGORITHM:
 *     1. Allocate data, indices, indptr arrays via registry
 *     2. Build indptr by cumulative sum of row/column lengths
 *     3. Copy values and indices sequentially
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(nnz + primary_dim) for output arrays
 *
 * THREAD SAFETY:
 *     Safe - read-only access to input matrix
 *
 * USAGE:
 *     auto arrs = to_contiguous_arrays(matrix);
 *     // Transfer ownership to Python before unregistering
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
ContiguousArraysT<T> to_contiguous_arrays(const Sparse<T, IsCSR>& matrix);

/* -----------------------------------------------------------------------------
 * FUNCTION: to_coo_arrays
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Export sparse matrix to COO (Coordinate) format with registry-registered
 *     arrays. Uses parallel processing for large matrices.
 *
 * PARAMETERS:
 *     matrix [in] Sparse matrix to export
 *
 * PRECONDITIONS:
 *     - matrix is valid
 *
 * POSTCONDITIONS:
 *     - Returns COOArraysT<T> with registry-registered pointers
 *     - row_indices[i], col_indices[i], values[i] form one non-zero entry
 *     - Entries are in row-major order (CSR) or column-major order (CSC)
 *     - If nnz == 0: all pointers are nullptr
 *     - Matrix unchanged
 *
 * RETURNS:
 *     COOArraysT<T> structure with registered pointers, or all-null on failure
 *
 * MUTABILITY:
 *     ALLOCATES - creates new registry-registered arrays
 *
 * ALGORITHM:
 *     1. Compute offsets for each primary slice
 *     2. Parallel conversion: each thread handles one or more primary slices
 *     3. Write COO triplets to pre-computed positions
 *
 * COMPLEXITY:
 *     Time:  O(nnz / n_threads + primary_dim)
 *     Space: O(nnz + primary_dim) for output arrays and offsets
 *
 * THREAD SAFETY:
 *     Safe - parallel writes to disjoint memory regions
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
COOArraysT<T> to_coo_arrays(const Sparse<T, IsCSR>& matrix);

/* -----------------------------------------------------------------------------
 * FUNCTION: from_contiguous_arrays
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Create sparse matrix from contiguous CSR/CSC format arrays. Arrays may
 *     be registry-registered or externally managed.
 *
 * PARAMETERS:
 *     data          [in] Pointer to values array
 *     indices       [in] Pointer to indices array
 *     indptr        [in] Pointer to offset array (size = primary_dim + 1)
 *     rows          [in] Number of rows
 *     cols          [in] Number of columns
 *     nnz           [in] Total number of non-zeros
 *     take_ownership [in] If true, register arrays with registry
 *
 * PRECONDITIONS:
 *     - data, indices, indptr are non-null
 *     - indptr[0] == 0, indptr[primary_dim] == nnz
 *     - indices are sorted within each row/column
 *     - All indices in valid range [0, secondary_dim)
 *
 * POSTCONDITIONS:
 *     - Returns Sparse matrix wrapping provided arrays
 *     - If take_ownership: arrays are registered with registry
 *     - If !take_ownership: zero-copy wrap (external ownership)
 *
 * RETURNS:
 *     Sparse matrix wrapping arrays, or empty matrix on failure
 *
 * MUTABILITY:
 *     CONST if !take_ownership, ALLOCATES if take_ownership
 *
 * COMPLEXITY:
 *     Time:  O(primary_dim) for metadata setup
 *     Space: O(primary_dim) for metadata arrays
 *
 * THREAD SAFETY:
 *     Safe - creates new matrix view
 *
 * WARNING:
 *     If take_ownership is false, caller must ensure arrays outlive the
 *     returned Sparse object.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Sparse<T, IsCSR> from_contiguous_arrays(
    T* data, Index* indices, Index* indptr,
    Index rows, Index cols, Index nnz,
    bool take_ownership = false
);

// =============================================================================
// Data Cleanup Tools
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: eliminate_zeros
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Remove zero-valued elements from sparse matrix, creating a new matrix
 *     with reduced storage. Uses parallel processing for counting and copying.
 *
 * PARAMETERS:
 *     matrix   [in] Input sparse matrix
 *     tolerance [in] Threshold for zero detection (default 0)
 *
 * PRECONDITIONS:
 *     - matrix is valid
 *
 * POSTCONDITIONS:
 *     - Returns new matrix with all |value| <= tolerance removed
 *     - Indices remain sorted
 *     - Original matrix unchanged
 *
 * RETURNS:
 *     New sparse matrix without zero elements
 *
 * MUTABILITY:
 *     ALLOCATES - creates new matrix
 *
 * ALGORITHM:
 *     1. Parallel count non-zeros per row/column after filtering
 *     2. Allocate new matrix with reduced nnz
 *     3. Parallel copy non-zero elements only
 *
 * COMPLEXITY:
 *     Time:  O(nnz / n_threads)
 *     Space: O(nnz_output + primary_dim) for result and counts
 *
 * THREAD SAFETY:
 *     Safe - parallel over independent rows/columns
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Sparse<T, IsCSR> eliminate_zeros(
    const Sparse<T, IsCSR>& matrix,
    T tolerance = T(0)
);

/* -----------------------------------------------------------------------------
 * FUNCTION: prune
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Remove small values from sparse matrix, optionally preserving structure.
 *
 * PARAMETERS:
 *     matrix        [in] Input sparse matrix
 *     threshold     [in] Values with |value| < threshold are removed/zeroed
 *     keep_structure [in] If true, set to zero but keep structure; if false, remove entirely
 *
 * PRECONDITIONS:
 *     - matrix is valid
 *
 * POSTCONDITIONS:
 *     - If keep_structure: all |value| < threshold set to zero, structure preserved
 *     - If !keep_structure: all |value| < threshold removed, structure compacted
 *     - Original matrix unchanged
 *
 * RETURNS:
 *     New sparse matrix with pruned values
 *
 * MUTABILITY:
 *     ALLOCATES - creates new matrix
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(nnz) for result (keep_structure) or O(nnz_output) (remove)
 *
 * THREAD SAFETY:
 *     Safe - read-only access to input
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Sparse<T, IsCSR> prune(
    const Sparse<T, IsCSR>& matrix,
    T threshold,
    bool keep_structure = false
);

// =============================================================================
// Validation and Info Tools
// =============================================================================

/* -----------------------------------------------------------------------------
 * STRUCT: ValidationResult
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Result structure for matrix validation.
 *
 * MEMBERS:
 *     valid         - true if matrix is valid
 *     error_message - Error description (nullptr if valid)
 *     error_index   - Index where error occurred (-1 if valid)
 * -------------------------------------------------------------------------- */
struct ValidationResult {
    bool valid;
    const char* error_message;  // nullptr if valid
    Index error_index;          // -1 if valid
};

/* -----------------------------------------------------------------------------
 * STRUCT: MemoryInfo
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Memory usage information for sparse matrix.
 *
 * MEMBERS:
 *     data_bytes     - Bytes used for values
 *     indices_bytes  - Bytes used for indices
 *     metadata_bytes - Bytes used for metadata (pointers, lengths)
 *     total_bytes    - Total memory usage
 *     block_count    - Number of allocated blocks
 *     is_contiguous  - Whether data is in single contiguous block
 * -------------------------------------------------------------------------- */
struct MemoryInfo {
    Size data_bytes;
    Size indices_bytes;
    Size metadata_bytes;
    Size total_bytes;
    Index block_count;
    bool is_contiguous;
};

/* -----------------------------------------------------------------------------
 * FUNCTION: validate
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Validate sparse matrix structure integrity: check indices are in range,
 *     sorted, and nnz consistency.
 *
 * PARAMETERS:
 *     matrix [in] Sparse matrix to validate
 *
 * PRECONDITIONS:
 *     None
 *
 * POSTCONDITIONS:
 *     - Returns ValidationResult with validation status
 *     - If invalid: error_message and error_index indicate problem
 *
 * RETURNS:
 *     ValidationResult structure
 *
 * ALGORITHM:
 *     1. Check matrix is valid (non-null pointers)
 *     2. For each row/column:
 *        - Check indices in range [0, secondary_dim)
 *        - Check indices are strictly ascending
 *     3. Verify total nnz matches sum of row/column lengths
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - read-only access
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
ValidationResult validate(const Sparse<T, IsCSR>& matrix);

/* -----------------------------------------------------------------------------
 * FUNCTION: memory_info
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Get detailed memory usage information for sparse matrix.
 *
 * PARAMETERS:
 *     matrix [in] Sparse matrix to analyze
 *
 * PRECONDITIONS:
 *     None
 *
 * POSTCONDITIONS:
 *     - Returns MemoryInfo with byte counts and layout information
 *
 * RETURNS:
 *     MemoryInfo structure
 *
 * COMPLEXITY:
 *     Time:  O(primary_dim) for block counting
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - read-only access
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
MemoryInfo memory_info(const Sparse<T, IsCSR>& matrix);

// =============================================================================
// Helper Conversion Tools
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: make_contiguous
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Convert sparse matrix to contiguous storage layout if not already.
 *
 * PARAMETERS:
 *     matrix [in] Input sparse matrix
 *
 * PRECONDITIONS:
 *     - matrix is valid
 *
 * POSTCONDITIONS:
 *     - Returns matrix with contiguous storage
 *     - Original matrix unchanged
 *
 * RETURNS:
 *     New sparse matrix with contiguous layout
 *
 * MUTABILITY:
 *     ALLOCATES - creates new matrix if not already contiguous
 *
 * COMPLEXITY:
 *     Time:  O(nnz) if conversion needed, O(primary_dim) if already contiguous
 *     Space: O(nnz) if conversion needed
 *
 * THREAD SAFETY:
 *     Safe - read-only access to input
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Sparse<T, IsCSR> make_contiguous(const Sparse<T, IsCSR>& matrix);

/* -----------------------------------------------------------------------------
 * FUNCTION: resize_secondary
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Resize secondary dimension of sparse matrix (metadata only, does not
 *     modify data or validate indices). In debug mode, validates indices when
 *     shrinking.
 *
 * PARAMETERS:
 *     matrix           [in,out] Sparse matrix to resize
 *     new_secondary_dim [in]     New secondary dimension size
 *
 * PRECONDITIONS:
 *     - matrix is valid
 *     - new_secondary_dim >= 0
 *     - If shrinking: all indices must be < new_secondary_dim
 *
 * POSTCONDITIONS:
 *     - Matrix secondary dimension updated
 *     - Data and indices unchanged (caller must ensure validity)
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix metadata
 *
 * COMPLEXITY:
 *     Time:  O(1) in release mode, O(nnz) in debug mode when shrinking
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Unsafe - modifies matrix state
 *
 * ASSERTIONS:
 *     Debug builds check all indices are in range when shrinking
 *
 * WARNING:
 *     This only updates dimension metadata. Caller must ensure all indices
 *     are valid for the new secondary dimension.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void resize_secondary(Sparse<T, IsCSR>& matrix, Index new_secondary_dim);

} // namespace scl::kernel::sparse
