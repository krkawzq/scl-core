// =============================================================================
// FILE: scl/kernel/comparison.h
// BRIEF: API reference for group comparison and differential abundance analysis
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::comparison {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_CELLS_PER_GROUP = 3;
    constexpr Size PERMUTATION_COUNT = 1000;
    constexpr Size PARALLEL_THRESHOLD = 32;
}

// =============================================================================
// Composition Analysis
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: composition_analysis
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Analyze cell type composition across conditions using chi-squared test.
 *
 * PARAMETERS:
 *     cell_types   [in]  Cell type labels for each cell [n_cells]
 *     conditions   [in]  Condition labels for each cell [n_cells]
 *     proportions  [out] Proportion matrix [n_types * n_conditions]
 *     p_values     [out] P-values for each cell type [n_types]
 *     n_types      [in]  Number of distinct cell types
 *     n_conditions [in]  Number of distinct conditions
 *
 * PRECONDITIONS:
 *     - cell_types.len == conditions.len
 *     - proportions has capacity >= n_types * n_conditions
 *     - p_values has capacity >= n_types
 *     - All cell type indices < n_types
 *     - All condition indices < n_conditions
 *
 * POSTCONDITIONS:
 *     - proportions[t * n_conditions + c] contains proportion of type t in condition c
 *     - p_values[t] contains chi-squared p-value for type t across conditions
 *     - All proportions are in [0, 1]
 *     - All p-values are in [0, 1]
 *
 * ALGORITHM:
 *     1. Count cells per type per condition
 *     2. Compute proportions for each type-condition pair
 *     3. For each type, compute chi-squared statistic comparing observed vs expected
 *     4. Convert chi-squared to p-value using Wilson-Hilferty approximation
 *
 * COMPLEXITY:
 *     Time:  O(n_cells + n_types * n_conditions)
 *     Space: O(n_types * n_conditions) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over types
 * -------------------------------------------------------------------------- */
void composition_analysis(
    Array<const Index> cell_types,        // Cell type labels [n_cells]
    Array<const Index> conditions,        // Condition labels [n_cells]
    Real* proportions,                    // Output proportions [n_types * n_conditions]
    Real* p_values,                       // Output p-values [n_types]
    Size n_types,                         // Number of cell types
    Size n_conditions                     // Number of conditions
);

/* -----------------------------------------------------------------------------
 * FUNCTION: abundance_test
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Test differential abundance of clusters between two conditions using
 *     Fisher's exact test approximation.
 *
 * PARAMETERS:
 *     cluster_labels [in]  Cluster assignment for each cell [n_cells]
 *     condition      [in]  Condition label (0 or 1) for each cell [n_cells]
 *     fold_changes   [out] Log2 fold change for each cluster [n_clusters]
 *     p_values       [out] P-values for each cluster [n_clusters]
 *
 * PRECONDITIONS:
 *     - cluster_labels.len == condition.len
 *     - fold_changes has capacity >= n_clusters
 *     - p_values has capacity >= n_clusters
 *     - Condition labels are 0 or 1
 *
 * POSTCONDITIONS:
 *     - fold_changes[c] contains log2(prop1 / prop0) for cluster c
 *     - p_values[c] contains Fisher's exact test p-value for cluster c
 *     - Fold changes may be infinite if one proportion is zero
 *
 * ALGORITHM:
 *     1. Count cells per cluster per condition
 *     2. Compute proportions for each cluster in each condition
 *     3. Compute fold change as log2(prop1 / prop0)
 *     4. Compute Fisher's exact test using chi-squared approximation
 *
 * COMPLEXITY:
 *     Time:  O(n_cells + n_clusters)
 *     Space: O(n_clusters) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over clusters
 * -------------------------------------------------------------------------- */
void abundance_test(
    Array<const Index> cluster_labels,    // Cluster labels [n_cells]
    Array<const Index> condition,         // Condition labels (0 or 1) [n_cells]
    Array<Real> fold_changes,             // Output fold changes [n_clusters]
    Array<Real> p_values                   // Output p-values [n_clusters]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: differential_abundance
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Test differential abundance across samples using Wilcoxon rank-sum test.
 *     Designed for multi-sample studies (e.g., DAseq, Milo).
 *
 * PARAMETERS:
 *     cluster_labels [in]  Cluster assignment for each cell [n_cells]
 *     sample_ids     [in]  Sample ID for each cell [n_cells]
 *     conditions     [in]  Condition label for each cell [n_cells]
 *     da_scores      [out] Differential abundance scores [n_clusters]
 *     p_values       [out] P-values from Wilcoxon test [n_clusters]
 *
 * PRECONDITIONS:
 *     - cluster_labels.len == sample_ids.len == conditions.len
 *     - da_scores has capacity >= n_clusters
 *     - p_values has capacity >= n_clusters
 *     - At least 2 conditions and 2 samples required
 *
 * POSTCONDITIONS:
 *     - da_scores[c] contains log2 fold change of mean proportions for cluster c
 *     - p_values[c] contains Wilcoxon rank-sum p-value for cluster c
 *     - Scores are computed comparing condition 0 vs condition 1
 *
 * ALGORITHM:
 *     1. Map samples to conditions
 *     2. Count cells per cluster per sample
 *     3. Compute proportions per sample
 *     4. For each cluster, collect proportions by condition
 *     5. Compute DA score as log2 fold change of means
 *     6. Perform Wilcoxon rank-sum test on proportions
 *
 * COMPLEXITY:
 *     Time:  O(n_cells + n_clusters * n_samples)
 *     Space: O(n_clusters * n_samples) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over clusters with workspace pools
 * -------------------------------------------------------------------------- */
void differential_abundance(
    Array<const Index> cluster_labels,    // Cluster labels [n_cells]
    Array<const Index> sample_ids,        // Sample IDs [n_cells]
    Array<const Index> conditions,        // Condition labels [n_cells]
    Array<Real> da_scores,                // Output DA scores [n_clusters]
    Array<Real> p_values                   // Output p-values [n_clusters]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: condition_response
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Test gene expression response between conditions using Wilcoxon test.
 *
 * PARAMETERS:
 *     expression       [in]  Expression matrix (cells x genes, CSR)
 *     conditions       [in]  Condition label for each cell [n_cells]
 *     response_scores  [out] Log2 fold change for each gene [n_genes]
 *     p_values         [out] P-values from Wilcoxon test [n_genes]
 *     n_genes          [in]  Number of genes
 *
 * PRECONDITIONS:
 *     - expression.rows() == conditions.len
 *     - response_scores has capacity >= n_genes
 *     - p_values has capacity >= n_genes
 *     - At least 2 conditions required
 *
 * POSTCONDITIONS:
 *     - response_scores[g] contains log2 fold change for gene g
 *     - p_values[g] contains Wilcoxon p-value for gene g
 *     - Scores compare condition 1 vs condition 0
 *
 * ALGORITHM:
 *     1. For each gene in parallel:
 *        a. Gather expression values for each condition using binary search
 *        b. Compute mean expression per condition
 *        c. Compute log2 fold change
 *        d. Perform Wilcoxon rank-sum test
 *
 * COMPLEXITY:
 *     Time:  O(n_genes * n_cells * log(nnz_per_cell))
 *     Space: O(n_cells) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over genes with workspace pools
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void condition_response(
    const Sparse<T, IsCSR>& expression,   // Expression matrix [n_cells x n_genes]
    Array<const Index> conditions,         // Condition labels [n_cells]
    Real* response_scores,                 // Output log2 fold changes [n_genes]
    Real* p_values,                         // Output p-values [n_genes]
    Size n_genes                            // Number of genes
);

/* -----------------------------------------------------------------------------
 * FUNCTION: effect_size
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Cohen's d effect size between two groups.
 *
 * PARAMETERS:
 *     group1 [in] Values for first group [n1]
 *     group2 [in] Values for second group [n2]
 *
 * PRECONDITIONS:
 *     - n1 >= 2 and n2 >= 2
 *
 * POSTCONDITIONS:
 *     - Returns Cohen's d = (mean2 - mean1) / pooled_sd
 *     - Returns 0.0 if pooled standard deviation is too small
 *
 * ALGORITHM:
 *     1. Compute means for both groups
 *     2. Compute variances for both groups
 *     3. Compute pooled standard deviation
 *     4. Return (mean2 - mean1) / pooled_sd
 *
 * COMPLEXITY:
 *     Time:  O(n1 + n2)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
Real effect_size(
    Array<const Real> group1,              // First group values [n1]
    Array<const Real> group2               // Second group values [n2]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: glass_delta
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Glass's delta effect size using control group standard deviation.
 *
 * PARAMETERS:
 *     control    [in] Control group values [n_control]
 *     treatment  [in] Treatment group values [n_treatment]
 *
 * PRECONDITIONS:
 *     - n_control >= 2 and n_treatment >= 1
 *
 * POSTCONDITIONS:
 *     - Returns Glass's delta = (mean_treatment - mean_control) / sd_control
 *     - Returns 0.0 if control standard deviation is too small
 *
 * ALGORITHM:
 *     1. Compute means for both groups
 *     2. Compute variance for control group
 *     3. Return (mean_treatment - mean_control) / sd_control
 *
 * COMPLEXITY:
 *     Time:  O(n_control + n_treatment)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
Real glass_delta(
    Array<const Real> control,             // Control group values [n_control]
    Array<const Real> treatment            // Treatment group values [n_treatment]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: hedges_g
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Hedges' g bias-corrected effect size.
 *
 * PARAMETERS:
 *     group1 [in] Values for first group [n1]
 *     group2 [in] Values for second group [n2]
 *
 * PRECONDITIONS:
 *     - n1 >= 2 and n2 >= 2
 *
 * POSTCONDITIONS:
 *     - Returns Hedges' g = d * correction_factor
 *     - Correction factor accounts for small sample bias
 *
 * ALGORITHM:
 *     1. Compute Cohen's d
 *     2. Apply bias correction: 1 - 3 / (4 * (n1 + n2) - 9)
 *     3. Return corrected effect size
 *
 * COMPLEXITY:
 *     Time:  O(n1 + n2)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
Real hedges_g(
    Array<const Real> group1,              // First group values [n1]
    Array<const Real> group2               // Second group values [n2]
);

} // namespace scl::kernel::comparison

