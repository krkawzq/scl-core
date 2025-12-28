// =============================================================================
// FILE: scl/kernel/qc.h
// BRIEF: API reference for quality control metrics with SIMD optimization
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::qc {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Real PCT_SCALE = Real(100);
}

// =============================================================================
// Transform Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_basic_qc
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute basic quality control metrics: number of genes and total counts
 *     per cell.
 *
 * PARAMETERS:
 *     matrix         [in]  Expression matrix (cells x genes, CSR)
 *     out_n_genes    [out] Number of expressed genes per cell [n_cells]
 *     out_total_counts [out] Total UMI counts per cell [n_cells]
 *
 * PRECONDITIONS:
 *     - out_n_genes.len == matrix.rows()
 *     - out_total_counts.len == matrix.rows()
 *     - Matrix must be valid CSR format
 *
 * POSTCONDITIONS:
 *     - out_n_genes[i] contains number of non-zero genes in cell i
 *     - out_total_counts[i] contains sum of all counts in cell i
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     For each cell in parallel:
 *         1. Count non-zero elements (number of genes)
 *         2. Sum all values using SIMD-optimized sum
 *         3. Write results to output arrays
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_basic_qc(
    const Sparse<T, IsCSR>& matrix,       // Expression matrix [n_cells x n_genes]
    Array<Index> out_n_genes,               // Output gene counts [n_cells]
    Array<Real> out_total_counts             // Output total counts [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_subset_pct
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute percentage of total counts that come from a subset of genes
 *     (e.g., mitochondrial genes) for each cell.
 *
 * PARAMETERS:
 *     matrix     [in]  Expression matrix (cells x genes, CSR)
 *     subset_mask [in]  Mask array, non-zero indicates subset gene [n_genes]
 *     out_pcts   [out] Percentage values [n_cells]
 *
 * PRECONDITIONS:
 *     - out_pcts.len == matrix.rows()
 *     - subset_mask.len >= matrix.cols()
 *     - Matrix must be valid CSR format
 *
 * POSTCONDITIONS:
 *     - out_pcts[i] contains percentage (0-100) of counts from subset in cell i
 *     - Returns 0.0 if total counts are zero
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     For each cell in parallel:
 *         1. Compute total counts and subset counts using fused SIMD operation
 *         2. Compute percentage = (subset / total) * 100
 *         3. Write result to output
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_subset_pct(
    const Sparse<T, IsCSR>& matrix,       // Expression matrix [n_cells x n_genes]
    Array<const uint8_t> subset_mask,      // Subset gene mask [n_genes]
    Array<Real> out_pcts                    // Output percentages [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_fused_qc
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute all QC metrics in a single pass: gene counts, total counts,
 *     and subset percentages.
 *
 * PARAMETERS:
 *     matrix         [in]  Expression matrix (cells x genes, CSR)
 *     subset_mask    [in]  Mask array for subset genes [n_genes]
 *     out_n_genes    [out] Number of expressed genes per cell [n_cells]
 *     out_total_counts [out] Total UMI counts per cell [n_cells]
 *     out_pcts       [out] Subset percentages per cell [n_cells]
 *
 * PRECONDITIONS:
 *     - All output arrays have length == matrix.rows()
 *     - subset_mask.len >= matrix.cols()
 *     - Matrix must be valid CSR format
 *
 * POSTCONDITIONS:
 *     - All metrics computed for each cell
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     For each cell in parallel:
 *         1. Count non-zero elements
 *         2. Compute total and subset counts using fused SIMD operation
 *         3. Compute percentage
 *         4. Write all results to output arrays
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_fused_qc(
    const Sparse<T, IsCSR>& matrix,       // Expression matrix [n_cells x n_genes]
    Array<const uint8_t> subset_mask,      // Subset gene mask [n_genes]
    Array<Index> out_n_genes,                // Output gene counts [n_cells]
    Array<Real> out_total_counts,            // Output total counts [n_cells]
    Array<Real> out_pcts                     // Output percentages [n_cells]
);

} // namespace scl::kernel::qc

