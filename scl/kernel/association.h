// =============================================================================
// FILE: scl/kernel/association.h
// BRIEF: API reference for feature association analysis across modalities (RNA + ATAC)
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
// Association Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: gene_peak_correlation
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute correlation between genes and accessible peaks.
 *
 * PARAMETERS:
 *     rna_expression [in]  RNA expression matrix (cells x genes, CSR)
 *     atac_peaks     [in]  ATAC peak matrix (cells x peaks, CSR)
 *     n_cells        [in]  Number of cells
 *     n_genes        [in]  Number of genes
 *     n_peaks        [in]  Number of peaks
 *     correlations   [out] Correlation matrix [n_genes * n_peaks]
 *
 * PRECONDITIONS:
 *     - correlations has capacity >= n_genes * n_peaks
 *
 * POSTCONDITIONS:
 *     - correlations[g * n_peaks + p] contains correlation
 *
 * COMPLEXITY:
 *     Time:  O(n_genes * n_peaks * n_cells)
 *     Space: O(n_cells) auxiliary per gene-peak pair
 *
 * THREAD SAFETY:
 *     Safe - parallelized over gene-peak pairs
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void gene_peak_correlation(
    const Sparse<T, IsCSR>& rna_expression,  // RNA expression [n_cells x n_genes]
    const Sparse<T, IsCSR>& atac_peaks,      // ATAC peaks [n_cells x n_peaks]
    Index n_cells,                           // Number of cells
    Index n_genes,                           // Number of genes
    Index n_peaks,                           // Number of peaks
    Real* correlations                        // Output correlations [n_genes * n_peaks]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: cis_regulatory_elements
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify cis-regulatory elements linked to genes.
 *
 * PARAMETERS:
 *     correlations    [in]  Gene-peak correlations [n_genes * n_peaks]
 *     peak_positions  [in]  Peak genomic positions [n_peaks * 2] (start, end)
 *     gene_positions  [in]  Gene positions [n_genes * 2] (start, end)
 *     n_genes         [in]  Number of genes
 *     n_peaks         [in]  Number of peaks
 *     max_distance    [in]  Maximum distance for cis-linkage
 *     linked_pairs    [out] Linked gene-peak pairs [max_results * 2]
 *     link_scores     [out] Link scores [max_results]
 *     max_results     [in]  Maximum number of results
 *
 * PRECONDITIONS:
 *     - linked_pairs has capacity >= max_results * 2
 *     - link_scores has capacity >= max_results
 *
 * POSTCONDITIONS:
 *     - Returns number of linked pairs found
 *     - linked_pairs[i * 2] = gene_index, linked_pairs[i * 2 + 1] = peak_index
 *
 * COMPLEXITY:
 *     Time:  O(n_genes * n_peaks)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over genes
 * -------------------------------------------------------------------------- */
Index cis_regulatory_elements(
    const Real* correlations,                // Gene-peak correlations [n_genes * n_peaks]
    const Index* peak_positions,              // Peak positions [n_peaks * 2]
    const Index* gene_positions,              // Gene positions [n_genes * 2]
    Index n_genes,                           // Number of genes
    Index n_peaks,                           // Number of peaks
    Index max_distance,                       // Max distance for cis-linkage
    Index* linked_pairs,                      // Output linked pairs [max_results * 2]
    Real* link_scores,                        // Output scores [max_results]
    Index max_results                         // Maximum results
);

} // namespace scl::kernel::association
