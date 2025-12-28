// =============================================================================
// FILE: scl/kernel/association.h
// BRIEF: API reference for feature association analysis across modalities
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::association {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Real MIN_CORRELATION = Real(0.1);
    constexpr Size MIN_CELLS_FOR_CORRELATION = 10;
    constexpr Size MAX_LINKS_PER_GENE = 1000;
    constexpr Size PARALLEL_THRESHOLD = 32;
}

// =============================================================================
// Gene-Peak Correlation
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: gene_peak_correlation
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Pearson correlation between gene expression and peak accessibility
 *     for all gene-peak pairs exceeding minimum correlation threshold.
 *
 * PARAMETERS:
 *     rna_expression      [in]  RNA expression matrix (cells x genes, CSR)
 *     atac_accessibility  [in]  ATAC accessibility matrix (cells x peaks, CSR)
 *     gene_indices        [out] Gene indices for significant correlations
 *     peak_indices        [out] Peak indices for significant correlations
 *     correlations        [out] Correlation values
 *     n_correlations      [out] Number of significant correlations found
 *     min_correlation     [in]  Minimum absolute correlation threshold
 *
 * PRECONDITIONS:
 *     - RNA and ATAC matrices must have same number of rows (cells)
 *     - Output arrays must have sufficient capacity for gene*peak pairs
 *
 * POSTCONDITIONS:
 *     - n_correlations pairs stored with |correlation| >= min_correlation
 *     - Input matrices are unchanged
 *
 * ALGORITHM:
 *     For each gene-peak pair:
 *         1. Gather expression and accessibility values using binary search
 *         2. Compute Pearson correlation
 *         3. If |correlation| >= threshold, store result
 *
 * COMPLEXITY:
 *     Time:  O(n_genes * n_peaks * n_cells * log(nnz_per_cell))
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses binary search for efficient feature lookup
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void gene_peak_correlation(
    const Sparse<T, IsCSR>& rna_expression,      // RNA expression [n_cells x n_genes]
    const Sparse<T, IsCSR>& atac_accessibility,  // ATAC accessibility [n_cells x n_peaks]
    Index* gene_indices,                          // Output gene indices [capacity]
    Index* peak_indices,                          // Output peak indices [capacity]
    Real* correlations,                           // Output correlations [capacity]
    Size& n_correlations,                         // Output: number of correlations found
    Real min_correlation                          // Minimum correlation threshold
);

// =============================================================================
// Cis-Regulatory Associations
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: cis_regulatory
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute correlations and p-values for specified gene-peak pairs,
 *     typically representing cis-regulatory associations.
 *
 * PARAMETERS:
 *     rna_expression      [in]  RNA expression matrix (cells x genes, CSR)
 *     atac_accessibility  [in]  ATAC accessibility matrix (cells x peaks, CSR)
 *     gene_indices        [in]  Gene indices for pairs to test
 *     peak_indices        [in]  Peak indices for pairs to test
 *     n_pairs             [in]  Number of gene-peak pairs
 *     correlations        [out] Computed correlations [n_pairs]
 *     p_values            [out] Computed p-values [n_pairs]
 *
 * PRECONDITIONS:
 *     - RNA and ATAC matrices must have same number of rows (cells)
 *     - Output arrays must have length >= n_pairs
 *     - All gene and peak indices must be valid
 *
 * POSTCONDITIONS:
 *     - correlations[i] contains Pearson correlation for pair i
 *     - p_values[i] contains two-tailed p-value for pair i
 *     - Input matrices are unchanged
 *
 * ALGORITHM:
 *     For each pair in parallel:
 *         1. Compute Pearson correlation
 *         2. Compute t-statistic: t = r * sqrt((n-2)/(1-r^2))
 *         3. Approximate p-value using normal approximation
 *
 * COMPLEXITY:
 *     Time:  O(n_pairs * n_cells * log(nnz_per_cell))
 *     Space: O(n_cells) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over pairs
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void cis_regulatory(
    const Sparse<T, IsCSR>& rna_expression,      // RNA expression [n_cells x n_genes]
    const Sparse<T, IsCSR>& atac_accessibility,  // ATAC accessibility [n_cells x n_peaks]
    const Index* gene_indices,                    // Gene indices [n_pairs]
    const Index* peak_indices,                    // Peak indices [n_pairs]
    Size n_pairs,                                 // Number of pairs
    Real* correlations,                           // Output correlations [n_pairs]
    Real* p_values                                // Output p-values [n_pairs]
);

// =============================================================================
// Enhancer-Gene Links
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: enhancer_gene_link
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify enhancer-gene links based on positive correlation between
 *     RNA expression and ATAC accessibility.
 *
 * PARAMETERS:
 *     rna                    [in]  RNA expression matrix (cells x genes, CSR)
 *     atac                   [in]  ATAC accessibility matrix (cells x peaks, CSR)
 *     correlation_threshold  [in]  Minimum positive correlation for link
 *     link_genes             [out] Gene indices for links
 *     link_peaks             [out] Peak indices for links
 *     link_correlations      [out] Correlation values for links
 *     n_links                [out] Number of links found
 *
 * PRECONDITIONS:
 *     - RNA and ATAC matrices must have same number of rows (cells)
 *     - Output arrays must have sufficient capacity
 *     - correlation_threshold >= MIN_CORRELATION
 *
 * POSTCONDITIONS:
 *     - n_links pairs stored with correlation >= threshold
 *     - At most MAX_LINKS_PER_GENE links per gene
 *     - Only positive correlations are included (enhancer-gene links)
 *     - Input matrices are unchanged
 *
 * ALGORITHM:
 *     For each gene:
 *         1. Iterate over peaks (up to MAX_LINKS_PER_GENE per gene)
 *         2. Compute Pearson correlation
 *         3. If correlation >= threshold, store link
 *
 * COMPLEXITY:
 *     Time:  O(n_genes * n_peaks * n_cells * log(nnz_per_cell))
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - binary search for efficient feature lookup
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void enhancer_gene_link(
    const Sparse<T, IsCSR>& rna,               // RNA expression [n_cells x n_genes]
    const Sparse<T, IsCSR>& atac,              // ATAC accessibility [n_cells x n_peaks]
    Real correlation_threshold,                 // Minimum correlation threshold
    Index* link_genes,                          // Output gene indices [capacity]
    Index* link_peaks,                          // Output peak indices [capacity]
    Real* link_correlations,                    // Output correlations [capacity]
    Size& n_links                               // Output: number of links found
);

// =============================================================================
// Multi-modal Neighbors
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: multimodal_neighbors
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute k-nearest neighbors using weighted combination of distances
 *     from two modalities.
 *
 * PARAMETERS:
 *     modality1           [in]  First modality data (cells x features, CSR)
 *     modality2           [in]  Second modality data (cells x features, CSR)
 *     weight1             [in]  Weight for first modality distance
 *     weight2             [in]  Weight for second modality distance
 *     k                   [in]  Number of neighbors to find
 *     neighbor_indices    [out] Neighbor indices [n_cells * k]
 *     neighbor_distances  [out] Neighbor distances [n_cells * k]
 *
 * PRECONDITIONS:
 *     - Both modalities must have same number of rows (cells)
 *     - Output arrays must have capacity >= n_cells * k
 *     - k <= n_cells - 1
 *
 * POSTCONDITIONS:
 *     - neighbor_indices[i*k + j] contains j-th nearest neighbor of cell i
 *     - neighbor_distances[i*k + j] contains distance to j-th neighbor
 *     - Distances computed as weight1*dist1 + weight2*dist2
 *     - Input matrices are unchanged
 *
 * ALGORITHM:
 *     For each cell in parallel:
 *         1. Compute combined distance to all other cells
 *         2. Use partial sort to find k smallest distances
 *         3. Store neighbor indices and distances
 *
 * COMPLEXITY:
 *     Time:  O(n_cells^2 * (nnz_per_cell + k * log(n_cells)))
 *     Space: O(n_threads * n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells with thread-local buffers
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void multimodal_neighbors(
    const Sparse<T, IsCSR>& modality1,         // First modality [n_cells x n_features1]
    const Sparse<T, IsCSR>& modality2,         // Second modality [n_cells x n_features2]
    Real weight1,                              // Weight for modality 1
    Real weight2,                              // Weight for modality 2
    Index k,                                   // Number of neighbors
    Index* neighbor_indices,                   // Output indices [n_cells * k]
    Real* neighbor_distances                   // Output distances [n_cells * k]
);

// =============================================================================
// Feature Coupling
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: feature_coupling
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify coupled features across modalities based on Spearman correlation.
 *
 * PARAMETERS:
 *     modality1           [in]  First modality data (cells x features, CSR)
 *     modality2           [in]  Second modality data (cells x features, CSR)
 *     feature1_indices    [out] Feature indices from modality 1
 *     feature2_indices    [out] Feature indices from modality 2
 *     coupling_scores     [out] Coupling scores (absolute Spearman correlation)
 *     n_couplings         [out] Number of couplings found
 *     min_score           [in]  Minimum coupling score threshold
 *
 * PRECONDITIONS:
 *     - Both modalities must have same number of rows (cells)
 *     - Output arrays must have sufficient capacity
 *
 * POSTCONDITIONS:
 *     - n_couplings pairs stored with |Spearman correlation| >= min_score
 *     - coupling_scores contains absolute Spearman correlation values
 *     - Input matrices are unchanged
 *
 * ALGORITHM:
 *     For each feature pair:
 *         1. Compute Spearman correlation using VQSort for ranking
 *         2. If |correlation| >= threshold, store coupling
 *
 * COMPLEXITY:
 *     Time:  O(n_features1 * n_features2 * n_cells * log(n_cells))
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses VQSort for efficient ranking
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void feature_coupling(
    const Sparse<T, IsCSR>& modality1,         // First modality [n_cells x n_features1]
    const Sparse<T, IsCSR>& modality2,         // Second modality [n_cells x n_features2]
    Index* feature1_indices,                   // Output feature1 indices [capacity]
    Index* feature2_indices,                   // Output feature2 indices [capacity]
    Real* coupling_scores,                     // Output coupling scores [capacity]
    Size& n_couplings,                         // Output: number of couplings found
    Real min_score                             // Minimum coupling score
);

// =============================================================================
// Correlation in Subset
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: correlation_in_subset
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Pearson correlation between two features in a subset of cells.
 *
 * PARAMETERS:
 *     data1           [in]  First data matrix (cells x features, CSR)
 *     feature1        [in]  Feature index in data1
 *     data2           [in]  Second data matrix (cells x features, CSR)
 *     feature2        [in]  Feature index in data2
 *     cell_indices    [in]  Indices of cells to include
 *     correlation     [out] Computed Pearson correlation
 *
 * PRECONDITIONS:
 *     - Both data matrices must have compatible dimensions
 *     - All cell indices must be valid
 *     - cell_indices.len >= MIN_CELLS_FOR_CORRELATION
 *
 * POSTCONDITIONS:
 *     - correlation contains Pearson correlation computed on subset
 *     - Returns 0 if subset too small or variance is near zero
 *     - Input matrices are unchanged
 *
 * ALGORITHM:
 *     1. Gather feature values for specified cells using binary search
 *     2. Compute mean, variance, and covariance
 *     3. Return correlation = cov / sqrt(var_x * var_y)
 *
 * COMPLEXITY:
 *     Time:  O(n_subset * log(nnz_per_cell))
 *     Space: O(n_subset) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - read-only access to matrices
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void correlation_in_subset(
    const Sparse<T, IsCSR>& data1,             // First data matrix
    Index feature1,                             // Feature index in data1
    const Sparse<T, IsCSR>& data2,             // Second data matrix
    Index feature2,                             // Feature index in data2
    Array<const Index> cell_indices,           // Cell indices to include
    Real& correlation                          // Output correlation
);

// =============================================================================
// Peak-to-Gene Activity
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: peak_to_gene_activity
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute gene activity scores by aggregating peak accessibility values
 *     based on peak-to-gene mapping.
 *
 * PARAMETERS:
 *     atac              [in]  ATAC accessibility matrix (cells x peaks, CSR)
 *     peak_to_gene_map  [in]  Mapping from peaks to genes (-1 for unmapped)
 *     n_peaks           [in]  Number of peaks
 *     n_genes           [in]  Number of genes
 *     gene_activity     [out] Gene activity matrix [n_cells * n_genes]
 *
 * PRECONDITIONS:
 *     - peak_to_gene_map has length >= n_peaks
 *     - gene_activity has capacity >= n_cells * n_genes
 *     - peak_to_gene_map values are in [-1, n_genes)
 *
 * POSTCONDITIONS:
 *     - gene_activity[c * n_genes + g] contains sum of accessibility
 *       for all peaks mapping to gene g in cell c
 *     - Peaks with mapping -1 are ignored
 *     - Input matrix is unchanged
 *
 * ALGORITHM:
 *     For each cell in parallel:
 *         1. Initialize gene activity to zero
 *         2. For each peak, add accessibility to mapped gene
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(n_cells * n_genes) for output
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells, each writes to independent memory
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void peak_to_gene_activity(
    const Sparse<T, IsCSR>& atac,              // ATAC accessibility [n_cells x n_peaks]
    const Index* peak_to_gene_map,              // Peak to gene mapping [n_peaks]
    Size n_peaks,                               // Number of peaks
    Size n_genes,                               // Number of genes
    Real* gene_activity                         // Output: [n_cells * n_genes]
);

} // namespace scl::kernel::association
