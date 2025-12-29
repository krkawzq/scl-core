#pragma once

// =============================================================================
// FILE: scl/binding/c_api/coexpression.h
// BRIEF: C API for co-expression network analysis (WGCNA-style)
// =============================================================================
//
// WEIGHTED GENE CO-EXPRESSION NETWORK ANALYSIS (WGCNA):
//   1. Compute gene-gene correlation matrix
//   2. Transform to weighted adjacency matrix (soft thresholding)
//   3. Compute topological overlap matrix (TOM)
//   4. Hierarchical clustering on TOM dissimilarity
//   5. Cut tree to identify modules (co-expressed gene groups)
//   6. Compute module eigengenes (first principal component)
//   7. Correlate module eigengenes with traits
//
// CORRELATION TYPES:
//   - Pearson: Linear correlation (default, fastest)
//   - Spearman: Rank correlation (robust to outliers)
//   - Bicor: Biweight midcorrelation (most robust)
//
// ADJACENCY TYPES:
//   - Unsigned: |corr|^power (all correlations positive)
//   - Signed: (0.5 + 0.5*corr)^power (preserves sign)
//   - SignedHybrid: |corr|^power * sign(corr)
//
// SOFT THRESHOLDING:
//   - Power typically in range [1, 20]
//   - Higher power = sparser network
//   - Choose power to achieve scale-free topology (R² > 0.8)
//
// OPTIMIZATIONS:
//   - SIMD-accelerated correlation computation
//   - Parallel upper-triangular matrix operations
//   - Cache-blocked gene expression extraction
//   - Optimized k-heap for hub gene identification
//
// MEMORY:
//   - Correlation matrix: n_genes² * 8 bytes
//   - For large n_genes (>10K), use blockwise processing
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Types
// =============================================================================

/// @brief Correlation method for co-expression analysis
// PERFORMANCE: C API uses typedef for C compatibility
// NOLINTNEXTLINE(modernize-use-using)
typedef enum {
    SCL_COEXPR_PEARSON = 0,      ///< Pearson correlation (fastest)
    SCL_COEXPR_SPEARMAN = 1,     ///< Spearman rank correlation
    SCL_COEXPR_BICOR = 2         ///< Biweight midcorrelation (most robust)
} scl_coexpr_correlation_t;

/// @brief Adjacency transformation type
// PERFORMANCE: C API uses typedef for C compatibility
// NOLINTNEXTLINE(modernize-use-using)
typedef enum {
    SCL_COEXPR_UNSIGNED = 0,         ///< Unsigned: |corr|^power
    SCL_COEXPR_SIGNED = 1,           ///< Signed: (0.5 + 0.5*corr)^power
    SCL_COEXPR_SIGNED_HYBRID = 2     ///< Signed hybrid: |corr|^power * sign
} scl_coexpr_adjacency_t;

// =============================================================================
// Correlation Matrix
// =============================================================================

/// @brief Compute gene-gene correlation matrix
/// @param[in] expression Expression matrix (cells x genes) (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_genes Number of genes
/// @param[out] corr_matrix Correlation matrix [n_genes * n_genes] (non-null)
/// @param[in] corr_type Correlation method
/// @return SCL_OK on success, error code otherwise
/// @note Output is symmetric with 1.0 on diagonal
/// @note Parallelized over gene pairs
scl_error_t scl_coexpr_correlation_matrix(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* corr_matrix,
    scl_coexpr_correlation_t corr_type
);

// =============================================================================
// WGCNA Adjacency Matrix
// =============================================================================

/// @brief Compute WGCNA weighted adjacency matrix
/// @param[in] expression Expression matrix (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_genes Number of genes
/// @param[in] power Soft threshold power
/// @param[out] adjacency Adjacency matrix [n_genes * n_genes] (non-null)
/// @param[in] corr_type Correlation method
/// @param[in] adj_type Adjacency transformation
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_coexpr_wgcna_adjacency(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t power,
    scl_real_t* adjacency,
    scl_coexpr_correlation_t corr_type,
    scl_coexpr_adjacency_t adj_type
);

// =============================================================================
// Topological Overlap Matrix (TOM)
// =============================================================================

/// @brief Compute topological overlap matrix from adjacency
/// @param[in] adjacency Adjacency matrix [n_genes * n_genes] (non-null)
/// @param[in] n_genes Number of genes
/// @param[out] tom TOM [n_genes * n_genes] (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note TOM measures shared neighbors between genes
scl_error_t scl_coexpr_topological_overlap(
    const scl_real_t* adjacency,
    scl_index_t n_genes,
    scl_real_t* tom
);

/// @brief Convert TOM to dissimilarity (1 - TOM)
/// @param[in] tom TOM [n_genes * n_genes] (non-null)
/// @param[in] n_genes Number of genes
/// @param[out] dissim Dissimilarity matrix [n_genes * n_genes] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_coexpr_tom_dissimilarity(
    const scl_real_t* tom,
    scl_index_t n_genes,
    scl_real_t* dissim
);

// =============================================================================
// Module Detection
// =============================================================================

/// @brief Detect co-expression modules via hierarchical clustering
/// @param[in] dissim Dissimilarity matrix [n_genes * n_genes] (non-null)
/// @param[in] n_genes Number of genes
/// @param[out] module_labels Module assignments [n_genes] (non-null)
/// @param[out] n_modules Number of modules detected (non-null)
/// @param[in] min_module_size Minimum genes per module (typical: 20-30)
/// @param[in] merge_cut_height Height to cut dendrogram (typical: 0.25)
/// @return SCL_OK on success, error code otherwise
/// @note Module 0 = unassigned (too small)
scl_error_t scl_coexpr_detect_modules(
    const scl_real_t* dissim,
    scl_index_t n_genes,
    scl_index_t* module_labels,
    scl_index_t* n_modules,
    scl_index_t min_module_size,
    scl_real_t merge_cut_height
);

// =============================================================================
// Module Eigengenes
// =============================================================================

/// @brief Compute module eigengene (first PC of module genes)
/// @param[in] expression Expression matrix (non-null)
/// @param[in] module_labels Module assignments [n_genes] (non-null)
/// @param[in] module_id Module ID to compute eigengene for
/// @param[in] n_cells Number of cells
/// @param[in] n_genes Number of genes
/// @param[out] eigengene Module eigengene [n_cells] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_coexpr_module_eigengene(
    scl_sparse_t expression,
    const scl_index_t* module_labels,
    scl_index_t module_id,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* eigengene
);

/// @brief Compute all module eigengenes
/// @param[in] expression Expression matrix (non-null)
/// @param[in] module_labels Module assignments [n_genes] (non-null)
/// @param[in] n_modules Number of modules
/// @param[in] n_cells Number of cells
/// @param[in] n_genes Number of genes
/// @param[out] eigengenes Eigengenes [n_cells * n_modules] (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Output: eigengenes[cell * n_modules + module]
scl_error_t scl_coexpr_all_eigengenes(
    scl_sparse_t expression,
    const scl_index_t* module_labels,
    scl_index_t n_modules,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* eigengenes
);

// =============================================================================
// Module-Trait Correlation
// =============================================================================

/// @brief Correlate module eigengenes with sample traits
/// @param[in] eigengenes Module eigengenes [n_samples * n_modules] (non-null)
/// @param[in] traits Sample traits [n_samples * n_traits] (non-null)
/// @param[in] n_samples Number of samples
/// @param[in] n_modules Number of modules
/// @param[in] n_traits Number of traits
/// @param[out] correlations Correlations [n_modules * n_traits] (non-null)
/// @param[out] p_values P-values [n_modules * n_traits] (nullable)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_coexpr_module_trait_correlation(
    const scl_real_t* eigengenes,
    const scl_real_t* traits,
    scl_index_t n_samples,
    scl_index_t n_modules,
    scl_index_t n_traits,
    scl_real_t* correlations,
    scl_real_t* p_values
);

// =============================================================================
// Hub Gene Identification
// =============================================================================

/// @brief Identify hub genes (highest intramodular connectivity)
/// @param[in] adjacency Adjacency matrix [n_genes * n_genes] (non-null)
/// @param[in] module_labels Module assignments [n_genes] (non-null)
/// @param[in] module_id Module to find hubs for
/// @param[in] n_genes Number of genes
/// @param[out] hub_genes Hub gene indices [max_hubs] (non-null)
/// @param[out] hub_scores Hub kME scores [max_hubs] (non-null)
/// @param[out] n_hubs Number of hub genes found (non-null)
/// @param[in] max_hubs Maximum hub genes to return
/// @return SCL_OK on success, error code otherwise
/// @note Hub genes have highest average connection within module
scl_error_t scl_coexpr_identify_hub_genes(
    const scl_real_t* adjacency,
    const scl_index_t* module_labels,
    scl_index_t module_id,
    scl_index_t n_genes,
    scl_index_t* hub_genes,
    scl_real_t* hub_scores,
    scl_index_t* n_hubs,
    scl_index_t max_hubs
);

// =============================================================================
// Gene-Module Membership (kME)
// =============================================================================

/// @brief Compute gene-module membership (correlation with eigengenes)
/// @param[in] expression Expression matrix (non-null)
/// @param[in] eigengenes Module eigengenes [n_cells * n_modules] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_genes Number of genes
/// @param[in] n_modules Number of modules
/// @param[out] kme_matrix kME matrix [n_genes * n_modules] (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note kME[gene, module] = correlation of gene with module eigengene
scl_error_t scl_coexpr_gene_module_membership(
    scl_sparse_t expression,
    const scl_real_t* eigengenes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_modules,
    scl_real_t* kme_matrix
);

// =============================================================================
// Soft Threshold Selection
// =============================================================================

/// @brief Pick optimal soft threshold power
/// @param[in] expression Expression matrix (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_genes Number of genes
/// @param[in] powers_to_test Array of powers to test [n_powers] (non-null)
/// @param[in] n_powers Number of powers to test
/// @param[out] scale_free_fits R² values for scale-free fit [n_powers] (non-null)
/// @param[out] mean_connectivity Mean connectivity [n_powers] (non-null)
/// @param[out] best_power Recommended power (non-null)
/// @param[in] corr_type Correlation method
/// @return SCL_OK on success, error code otherwise
/// @note Choose power where R² > 0.8 (scale-free topology)
scl_error_t scl_coexpr_pick_soft_threshold(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    const scl_real_t* powers_to_test,
    scl_index_t n_powers,
    scl_real_t* scale_free_fits,
    scl_real_t* mean_connectivity,
    scl_real_t* best_power,
    scl_coexpr_correlation_t corr_type
);

// =============================================================================
// Blockwise Module Detection
// =============================================================================

/// @brief Detect modules in large networks via blockwise processing
/// @param[in] expression Expression matrix (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_genes Number of genes
/// @param[in] block_size Genes per block (typical: 5000)
/// @param[in] power Soft threshold power
/// @param[in] min_module_size Minimum genes per module
/// @param[out] module_labels Module assignments [n_genes] (non-null)
/// @param[out] n_modules Number of modules detected (non-null)
/// @param[in] corr_type Correlation method
/// @return SCL_OK on success, error code otherwise
/// @note Use for n_genes > 10000 to reduce memory usage
scl_error_t scl_coexpr_blockwise_modules(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t block_size,
    scl_real_t power,
    scl_index_t min_module_size,
    scl_index_t* module_labels,
    scl_index_t* n_modules,
    scl_coexpr_correlation_t corr_type
);

// =============================================================================
// Module Preservation
// =============================================================================

/// @brief Assess module preservation between datasets
/// @param[in] adjacency_ref Reference adjacency [n_genes * n_genes] (non-null)
/// @param[in] adjacency_test Test adjacency [n_genes * n_genes] (non-null)
/// @param[in] module_labels Module assignments [n_genes] (non-null)
/// @param[in] n_genes Number of genes
/// @param[in] n_modules Number of modules
/// @param[out] zsummary Z-summary scores [n_modules] (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Z > 2: weak preservation, Z > 10: strong preservation
scl_error_t scl_coexpr_module_preservation(
    const scl_real_t* adjacency_ref,
    const scl_real_t* adjacency_test,
    const scl_index_t* module_labels,
    scl_index_t n_genes,
    scl_index_t n_modules,
    scl_real_t* zsummary
);

#ifdef __cplusplus
}
#endif
