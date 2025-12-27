#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/association.hpp
// BRIEF: Feature association across modalities (RNA + ATAC)
//
// APPLICATIONS:
// - Gene-peak correlation
// - Cis-regulatory element identification
// - Enhancer-gene linking
// =============================================================================

namespace scl::kernel::association {

// TODO: Gene-peak correlation
template <typename T, bool IsCSR>
void gene_peak_correlation(
    const Sparse<T, IsCSR>& rna_expression,
    const Sparse<T, IsCSR>& atac_accessibility,
    Sparse<Real, true>& correlation_matrix
);

// TODO: Cis-regulatory associations
template <typename T, bool IsCSR>
void cis_regulatory(
    const Sparse<T, IsCSR>& rna_expression,
    const Sparse<T, IsCSR>& atac_accessibility,
    const std::vector<std::pair<Index, Index>>& gene_peak_pairs,  // Within distance
    Sparse<Real, true>& cis_associations
);

// TODO: Enhancer-gene links
template <typename T, bool IsCSR>
void enhancer_gene_link(
    const Sparse<T, IsCSR>& rna,
    const Sparse<T, IsCSR>& atac,
    Real correlation_threshold,
    std::vector<std::pair<Index, Index>>& links
);

// TODO: Multi-modal neighbors
template <typename T, bool IsCSR>
void multimodal_neighbors(
    const Sparse<T, IsCSR>& modality1,
    const Sparse<T, IsCSR>& modality2,
    const std::vector<Real>& weights,
    Index k,
    Sparse<Index, IsCSR>& neighbors
);

// TODO: Feature coupling
template <typename T, bool IsCSR>
void feature_coupling(
    const Sparse<T, IsCSR>& modality1,
    const Sparse<T, IsCSR>& modality2,
    Sparse<Real, true>& coupling_scores
);

} // namespace scl::kernel::association

