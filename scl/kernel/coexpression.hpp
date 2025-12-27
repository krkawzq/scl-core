#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/coexpression.hpp
// BRIEF: Co-expression module detection (WGCNA-style)
//
// APPLICATIONS:
// - Gene module detection
// - Module-trait correlation
// - Hub gene identification
// =============================================================================

namespace scl::kernel::coexpression {

// TODO: WGCNA adjacency matrix
template <typename T, bool IsCSR>
void wgcna_adjacency(
    const Sparse<T, IsCSR>& expression,
    Real power,
    Sparse<Real, true>& adjacency
);

// TODO: Module detection
template <typename T, bool IsCSR>
void detect_modules(
    const Sparse<Real, true>& adjacency,
    Index min_module_size,
    Array<Index> module_labels
);

// TODO: Module eigengene
template <typename T, bool IsCSR>
void module_eigengene(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> module_labels,
    Index module_id,
    Array<Real> eigengene
);

// TODO: Hub gene identification
void identify_hub_genes(
    const Sparse<Real, true>& adjacency,
    Array<const Index> module_labels,
    Index module_id,
    std::vector<Index>& hub_genes
);

} // namespace scl::kernel::coexpression

