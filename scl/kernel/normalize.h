// =============================================================================
// FILE: scl/kernel/normalize.h
// BRIEF: API reference for normalization operations with SIMD optimization
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::normalize {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 64;
}

// =============================================================================
// Transform Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_row_sums
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute sum of values in each row of a sparse matrix.
 *
 * PARAMETERS:
 *     matrix [in]  Sparse matrix (CSR or CSC)
 *     output [out] Pre-allocated buffer for row sums [n_rows]
 *
 * PRECONDITIONS:
 *     - output.len >= matrix.rows()
 *     - Matrix must be valid sparse format
 *
 * POSTCONDITIONS:
 *     - output[i] contains sum of row i
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     For each row in parallel:
 *         1. Iterate over non-zero elements in row
 *         2. Sum values using vectorized sum
 *         3. Write result to output
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_row_sums(
    const Sparse<T, IsCSR>& matrix,       // Sparse matrix input
    Array<T> output                        // Output row sums [n_rows]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: scale_primary
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Scale each primary dimension (row for CSR, column for CSC) by a factor.
 *
 * PARAMETERS:
 *     matrix [in,out] Sparse matrix, modified in-place
 *     scales [in]     Scaling factors [primary_dim]
 *
 * PRECONDITIONS:
 *     - scales.len >= matrix.primary_dim()
 *     - Matrix values must be mutable
 *
 * POSTCONDITIONS:
 *     - Each primary dimension is scaled by corresponding factor
 *     - Matrix structure (indices, indptr) unchanged
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * ALGORITHM:
 *     For each primary dimension in parallel:
 *         1. If scale == 1.0, skip
 *         2. Load values for this dimension
 *         3. Scale using SIMD operations
 *         4. Store scaled values back
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimensions
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void scale_primary(
    Sparse<T, IsCSR>& matrix,              // Sparse matrix, modified in-place
    Array<const Real> scales                // Scaling factors [primary_dim]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: primary_sums_masked
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute sum of values in each primary dimension, counting only elements
 *     where mask[indices[i]] == 0.
 *
 * PARAMETERS:
 *     matrix [in]  Sparse matrix (CSR or CSC)
 *     mask   [in]  Mask array, 0 indicates element should be counted [secondary_dim]
 *     output [out] Pre-allocated buffer for sums [primary_dim]
 *
 * PRECONDITIONS:
 *     - output.len >= matrix.primary_dim()
 *     - mask.len >= matrix.secondary_dim()
 *     - Matrix must be valid sparse format
 *
 * POSTCONDITIONS:
 *     - output[i] contains sum of unmasked elements in primary dimension i
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     For each primary dimension in parallel:
 *         1. Iterate over non-zero elements
 *         2. Check mask[indices[j]] == 0
 *         3. Sum unmasked values using optimized SIMD path
 *         4. Write result to output
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimensions
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void primary_sums_masked(
    const Sparse<T, IsCSR>& matrix,       // Sparse matrix input
    Array<const Byte> mask,                 // Mask array [secondary_dim]
    Array<Real> output                      // Output sums [primary_dim]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: detect_highly_expressed
 * -----------------------------------------------------------------------------
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Detect genes that are highly expressed in each cell, where expression
 *     exceeds a fraction of total expression.
 *
 * PARAMETERS:
 *     matrix       [in]  Expression matrix (cells x genes, CSR)
 *     row_sums     [in]  Pre-computed row sums [n_cells]
 *     max_fraction [in]  Maximum fraction of total expression per gene
 *     out_mask     [out] Output mask, 1 indicates highly expressed [n_genes]
 *
 * PRECONDITIONS:
 *     - row_sums.len == matrix.rows()
 *     - out_mask.len >= matrix.cols()
 *     - max_fraction in (0, 1]
 *
 * POSTCONDITIONS:
 *     - out_mask[g] == 1 if gene g is highly expressed in any cell
 *     - out_mask[g] == 0 otherwise
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     For each cell in parallel:
 *         1. Compute threshold = row_sums[cell] * max_fraction
 *         2. For each expressed gene in cell:
 *            a. If value > threshold, set out_mask[gene] = 1 (atomic)
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses atomic operations for mask updates
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void detect_highly_expressed(
    const Sparse<T, IsCSR>& matrix,       // Expression matrix [n_cells x n_genes]
    Array<const Real> row_sums,             // Pre-computed row sums [n_cells]
    Real max_fraction,                       // Maximum fraction threshold
    Array<Byte> out_mask                     // Output mask [n_genes]
);

} // namespace scl::kernel::normalize

