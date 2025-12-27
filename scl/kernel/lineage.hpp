#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/lineage.hpp
// BRIEF: Lineage tracing and fate mapping
//
// APPLICATIONS:
// - Developmental lineage reconstruction
// - Fate bias quantification
// - Barcode analysis
// =============================================================================

namespace scl::kernel::lineage {

// TODO: Lineage tree construction
void lineage_tree(
    const std::vector<std::vector<Index>>& clones,
    std::vector<std::pair<Index, Index>>& tree_edges
);

// TODO: Lineage coupling
void lineage_coupling(
    Array<const Index> clone_ids,
    Array<const Index> cell_types,
    Sparse<Real, true>& coupling_matrix
);

// TODO: Fate bias
void fate_bias(
    Array<const Index> clone_ids,
    Array<const Index> cell_types,
    Array<Real> bias_scores
);

// TODO: Lineage distance
void lineage_distance(
    const std::vector<std::pair<Index, Index>>& tree_edges,
    Sparse<Real, true>& distance_matrix
);

// TODO: Barcode analysis
void barcode_analysis(
    const std::vector<std::string>& barcodes,
    std::vector<Index>& clone_ids,
    Size& n_unique_clones
);

} // namespace scl::kernel::lineage

