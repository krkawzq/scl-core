// =============================================================================
// FILE: scl/kernel/outlier.h
// BRIEF: API reference for outlier and anomaly detection kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::outlier {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_K_NEIGHBORS = 5;
    constexpr Size DEFAULT_K = 20;
    constexpr Real LOF_THRESHOLD = Real(1.5);
    constexpr Real AMBIENT_THRESHOLD = Real(0.1);
    constexpr Size EMPTY_DROPS_MIN_UMI = 100;
    constexpr Size EMPTY_DROPS_MAX_AMBIENT = 10;
    constexpr Size MONTE_CARLO_ITERATIONS = 10000;
    constexpr Size PARALLEL_THRESHOLD = 256;
}

// =============================================================================
// Isolation Score
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: isolation_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute isolation scores based on statistical deviation from global
 *     cell population characteristics.
 *
 * PARAMETERS:
 *     data   [in]  Expression matrix (cells x genes, CSR)
 *     scores [out] Per-cell isolation scores
 *
 * PRECONDITIONS:
 *     - data.rows() == scores.len
 *     - scores array is pre-allocated
 *
 * POSTCONDITIONS:
 *     - scores[i] >= 0, higher values indicate more isolated cells
 *     - Score is average of mean deviation and variance deviation
 *
 * ALGORITHM:
 *     1. Compute per-cell mean and variance over all genes
 *     2. Compute global mean and variance
 *     3. Score = (abs(cell_mean - global_mean)/global_std +
 *                 abs(cell_var - global_var)/global_var) / 2
 *
 * COMPLEXITY:
 *     Time:  O(nnz + n_cells * n_features)
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void isolation_score(
    const Sparse<T, IsCSR>& data,        // Expression matrix
    Array<Real> scores                    // Output isolation scores [n_cells]
);

// =============================================================================
// Local Outlier Factor (LOF)
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: local_outlier_factor
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Local Outlier Factor for each cell based on local density
 *     compared to neighbors.
 *
 * PARAMETERS:
 *     data       [in]  Expression matrix
 *     neighbors  [in]  KNN neighbor indices (CSR)
 *     distances  [in]  KNN distances (CSR)
 *     lof_scores [out] Per-cell LOF scores
 *
 * PRECONDITIONS:
 *     - All matrices have same number of rows
 *     - neighbors and distances are aligned (same structure)
 *     - lof_scores.len == data.rows()
 *
 * POSTCONDITIONS:
 *     - lof_scores[i] >= 0
 *     - LOF ~ 1 indicates normal density
 *     - LOF > 1.5 typically indicates outlier
 *
 * ALGORITHM:
 *     1. Compute k-distance for each point (distance to k-th neighbor)
 *     2. Compute local reachability density (LRD) using reachability distance
 *     3. LOF = mean(LRD_neighbors) / LRD_point
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * k^2) where k = neighbors per cell
 *     Space: O(n_cells + k) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void local_outlier_factor(
    const Sparse<T, IsCSR>& data,           // Expression matrix
    const Sparse<Index, IsCSR>& neighbors,  // KNN indices
    const Sparse<Real, IsCSR>& distances,   // KNN distances
    Array<Real> lof_scores                   // Output LOF scores [n_cells]
);

// =============================================================================
// Ambient RNA Detection
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: ambient_detection
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute ambient RNA contamination score for each cell based on
 *     correlation with estimated ambient profile.
 *
 * PARAMETERS:
 *     expression     [in]  Expression matrix (cells x genes)
 *     ambient_scores [out] Per-cell ambient contamination scores
 *
 * PRECONDITIONS:
 *     - expression.rows() == ambient_scores.len
 *
 * POSTCONDITIONS:
 *     - ambient_scores[i] in [0, 1]
 *     - 1 indicates high correlation with ambient profile
 *     - 0 indicates no correlation
 *
 * ALGORITHM:
 *     1. Compute total UMI per cell
 *     2. Identify bottom 10% UMI cells as ambient reference
 *     3. Build ambient profile from reference cells
 *     4. Compute cosine similarity between each cell and ambient profile
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_genes + nnz)
 *     Space: O(n_cells + n_genes) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 *
 * NUMERICAL NOTES:
 *     Uses VQSort for efficient UMI sorting
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void ambient_detection(
    const Sparse<T, IsCSR>& expression,  // Expression matrix
    Array<Real> ambient_scores            // Output ambient scores [n_cells]
);

// =============================================================================
// Empty Droplet Detection
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: empty_drops
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify empty droplets using deviance test against ambient profile
 *     (EmptyDrops-style algorithm).
 *
 * PARAMETERS:
 *     raw_counts    [in]  Raw UMI count matrix (cells x genes)
 *     is_empty      [out] Boolean mask for empty droplets
 *     fdr_threshold [in]  FDR threshold for calling cells
 *
 * PRECONDITIONS:
 *     - raw_counts.rows() == is_empty.len
 *     - fdr_threshold in (0, 1)
 *
 * POSTCONDITIONS:
 *     - is_empty[i] = true if cell i is likely empty
 *     - Cells with UMI < EMPTY_DROPS_MIN_UMI marked as empty
 *
 * ALGORITHM:
 *     1. Sort cells by total UMI
 *     2. Estimate ambient profile from lowest-UMI barcodes
 *     3. For each cell above minimum UMI:
 *        a. Compute deviance from ambient expectation
 *        b. P-value from chi-squared distribution
 *     4. Apply BH correction for FDR control
 *     5. Mark cells failing test as empty
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_genes + n_cells log n_cells)
 *     Space: O(n_cells + n_genes) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 *
 * NUMERICAL NOTES:
 *     Uses Wilson-Hilferty approximation for chi-squared CDF
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void empty_drops(
    const Sparse<T, IsCSR>& raw_counts,  // Raw count matrix
    Array<bool> is_empty,                 // Output empty mask [n_cells]
    Real fdr_threshold = Real(0.01)       // FDR threshold
);

// =============================================================================
// Outlier Gene Detection
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: outlier_genes
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify genes with outlier dispersion characteristics based on
 *     median absolute deviation of log CV^2.
 *
 * PARAMETERS:
 *     expression          [in]  Expression matrix
 *     outlier_gene_indices [out] Indices of outlier genes
 *     n_outliers          [out] Number of outliers found
 *     threshold           [in]  Z-score threshold for outlier detection
 *
 * PRECONDITIONS:
 *     - outlier_gene_indices has capacity >= expression.cols()
 *     - threshold > 0
 *
 * POSTCONDITIONS:
 *     - n_outliers = number of outlier genes found
 *     - outlier_gene_indices[0..n_outliers) contains outlier gene indices
 *
 * ALGORITHM:
 *     1. Compute mean and variance for each gene
 *     2. Compute log CV^2 for genes with mean > epsilon
 *     3. Find median and MAD of log CV^2
 *     4. Flag genes with z-score > threshold
 *
 * COMPLEXITY:
 *     Time:  O(nnz + n_genes log n_genes)
 *     Space: O(n_genes) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 *
 * NUMERICAL NOTES:
 *     MAD scaled by 1.4826 for normal distribution consistency
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void outlier_genes(
    const Sparse<T, IsCSR>& expression,  // Expression matrix
    Index* outlier_gene_indices,          // Output outlier indices
    Size& n_outliers,                     // Output count
    Real threshold = Real(3.0)            // Z-score threshold
);

// =============================================================================
// Doublet Score
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: doublet_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute doublet scores based on expression dissimilarity from
 *     local neighborhood.
 *
 * PARAMETERS:
 *     expression [in]  Expression matrix
 *     neighbors  [in]  KNN graph
 *     scores     [out] Per-cell doublet scores
 *
 * PRECONDITIONS:
 *     - expression.rows() == neighbors.rows() == scores.len
 *
 * POSTCONDITIONS:
 *     - scores[i] >= 0
 *     - Higher scores indicate potential doublets
 *
 * ALGORITHM:
 *     For each cell:
 *         1. Compute variance of expression across neighbors per feature
 *         2. Compute z-score of cell value vs neighbor distribution
 *         3. Score = mean z-score across features
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * k * nnz_per_row) where k = neighbors per cell
 *     Space: O(1) auxiliary per cell
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void doublet_score(
    const Sparse<T, IsCSR>& expression,     // Expression matrix
    const Sparse<Index, IsCSR>& neighbors,  // KNN graph
    Array<Real> scores                       // Output scores [n_cells]
);

// =============================================================================
// Mitochondrial Outliers
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: mitochondrial_outliers
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify cells with high mitochondrial gene content, typically
 *     indicating damaged or dying cells.
 *
 * PARAMETERS:
 *     expression    [in]  Expression matrix
 *     mito_genes    [in]  Indices of mitochondrial genes
 *     mito_fraction [out] Mitochondrial fraction per cell
 *     is_outlier    [out] Boolean mask for high-mito cells
 *     threshold     [in]  Fraction threshold for outlier status
 *
 * PRECONDITIONS:
 *     - All arrays sized to expression.rows()
 *     - mito_genes indices within expression.cols() range
 *     - threshold in (0, 1)
 *
 * POSTCONDITIONS:
 *     - mito_fraction[i] = mito_UMI / total_UMI for cell i
 *     - is_outlier[i] = (mito_fraction[i] > threshold)
 *
 * COMPLEXITY:
 *     Time:  O(nnz + n_cells)
 *     Space: O(max_gene_idx) for gene lookup
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void mitochondrial_outliers(
    const Sparse<T, IsCSR>& expression,  // Expression matrix
    Array<const Index> mito_genes,        // Mitochondrial gene indices
    Array<Real> mito_fraction,            // Output mito fraction [n_cells]
    Array<bool> is_outlier,               // Output outlier mask [n_cells]
    Real threshold = Real(0.2)            // Fraction threshold
);

// =============================================================================
// QC Filter
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: qc_filter
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply combined QC filtering based on gene count, UMI count, and
 *     mitochondrial fraction thresholds.
 *
 * PARAMETERS:
 *     expression        [in]  Expression matrix
 *     min_genes         [in]  Minimum detected genes per cell
 *     max_genes         [in]  Maximum detected genes per cell
 *     min_counts        [in]  Minimum total counts per cell
 *     max_counts        [in]  Maximum total counts per cell
 *     max_mito_fraction [in]  Maximum mitochondrial fraction
 *     mito_genes        [in]  Mitochondrial gene indices
 *     pass_qc           [out] Boolean mask for passing cells
 *
 * PRECONDITIONS:
 *     - pass_qc.len == expression.rows()
 *     - min_genes <= max_genes
 *     - min_counts <= max_counts
 *     - max_mito_fraction in [0, 1]
 *
 * POSTCONDITIONS:
 *     - pass_qc[i] = true if cell passes all QC criteria
 *
 * ALGORITHM:
 *     For each cell:
 *         1. Count detected genes (non-zero entries)
 *         2. Sum total counts
 *         3. Compute mitochondrial fraction
 *         4. Apply all thresholds
 *
 * COMPLEXITY:
 *     Time:  O(nnz + n_cells)
 *     Space: O(max_gene_idx) for mito gene lookup
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void qc_filter(
    const Sparse<T, IsCSR>& expression,  // Expression matrix
    Real min_genes,                       // Min genes per cell
    Real max_genes,                       // Max genes per cell
    Real min_counts,                      // Min UMI per cell
    Real max_counts,                      // Max UMI per cell
    Real max_mito_fraction,               // Max mito fraction
    Array<const Index> mito_genes,        // Mitochondrial gene indices
    Array<bool> pass_qc                   // Output QC pass mask [n_cells]
);

} // namespace scl::kernel::outlier
