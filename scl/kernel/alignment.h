// =============================================================================
// FILE: scl/kernel/alignment.h
// BRIEF: API reference for multi-modal data alignment and batch integration
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::alignment {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size DEFAULT_K = 30;
    constexpr Real ANCHOR_SCORE_THRESHOLD = Real(0.5);
    constexpr Size MAX_ANCHORS_PER_CELL = 10;
    constexpr Size PARALLEL_THRESHOLD = 32;
}

// =============================================================================
// Alignment Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: find_anchors
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Find anchor pairs between two datasets (Seurat-style integration).
 *
 * PARAMETERS:
 *     query_data    [in]  Query expression matrix (cells x genes, CSR)
 *     reference_data [in]  Reference expression matrix (cells x genes, CSR)
 *     n_query       [in]  Number of query cells
 *     n_ref         [in]  Number of reference cells
 *     k             [in]  Number of neighbors for anchor scoring
 *     anchor_pairs  [out] Anchor pairs [max_anchors * 2]
 *     anchor_scores [out] Anchor scores [max_anchors]
 *     max_anchors   [in]  Maximum number of anchors
 *
 * PRECONDITIONS:
 *     - anchor_pairs has capacity >= max_anchors * 2
 *     - anchor_scores has capacity >= max_anchors
 *
 * POSTCONDITIONS:
 *     - Returns number of anchors found
 *     - anchor_pairs[i * 2] = query_idx, anchor_pairs[i * 2 + 1] = ref_idx
 *
 * COMPLEXITY:
 *     Time:  O(n_query * n_ref * log(n_ref))
 *     Space: O(n_query * k) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over query cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Index find_anchors(
    const Sparse<T, IsCSR>& query_data,     // Query expression [n_query x n_genes]
    const Sparse<T, IsCSR>& reference_data,  // Reference expression [n_ref x n_genes]
    Index n_query,                           // Number of query cells
    Index n_ref,                             // Number of reference cells
    Size k,                                  // Number of neighbors
    Index* anchor_pairs,                      // Output anchor pairs [max_anchors * 2]
    Real* anchor_scores,                     // Output anchor scores [max_anchors]
    Index max_anchors                         // Maximum anchors
);

/* -----------------------------------------------------------------------------
 * FUNCTION: mnn_correction
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply mutual nearest neighbors (MNN) batch correction.
 *
 * PARAMETERS:
 *     query_data    [in,out] Query expression matrix, modified in-place
 *     reference_data [in]  Reference expression matrix
 *     mnn_pairs      [in]  MNN pairs [n_mnn * 2]
 *     n_mnn          [in]  Number of MNN pairs
 *     n_genes        [in]  Number of genes
 *
 * PRECONDITIONS:
 *     - query_data values are mutable
 *
 * POSTCONDITIONS:
 *     - query_data is corrected toward reference
 *
 * MUTABILITY:
 *     INPLACE - modifies query_data.values()
 *
 * COMPLEXITY:
 *     Time:  O(n_mnn * n_genes)
 *     Space: O(n_genes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over MNN pairs
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void mnn_correction(
    Sparse<T, IsCSR>& query_data,            // Query expression [n_query x n_genes]
    const Sparse<T, IsCSR>& reference_data,  // Reference expression [n_ref x n_genes]
    const Index* mnn_pairs,                  // MNN pairs [n_mnn * 2]
    Index n_mnn,                             // Number of MNN pairs
    Index n_genes                             // Number of genes
);

/* -----------------------------------------------------------------------------
 * FUNCTION: integration_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute integration quality score between datasets.
 *
 * PARAMETERS:
 *     query_data    [in]  Query expression matrix (cells x genes, CSR)
 *     reference_data [in]  Reference expression matrix (cells x genes, CSR)
 *     n_query       [in]  Number of query cells
 *     n_ref         [in]  Number of reference cells
 *     k             [in]  Number of neighbors for scoring
 *     score         [out] Integration score
 *
 * PRECONDITIONS:
 *     - Both datasets have same number of genes
 *
 * POSTCONDITIONS:
 *     - score contains integration quality (higher is better)
 *
 * COMPLEXITY:
 *     Time:  O((n_query + n_ref) * k * log(n_ref))
 *     Space: O(k) auxiliary per cell
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void integration_score(
    const Sparse<T, IsCSR>& query_data,     // Query expression [n_query x n_genes]
    const Sparse<T, IsCSR>& reference_data,  // Reference expression [n_ref x n_genes]
    Index n_query,                           // Number of query cells
    Index n_ref,                             // Number of reference cells
    Size k,                                  // Number of neighbors
    Real& score                               // Output integration score
);

} // namespace scl::kernel::alignment
