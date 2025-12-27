#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/enrichment.hpp
// BRIEF: Gene set enrichment and pathway analysis
//
// APPLICATIONS:
// - GSEA
// - Over-representation analysis
// - Pathway scoring
// =============================================================================

namespace scl::kernel::enrichment {

// TODO: Hypergeometric test
Real hypergeometric_test(
    Index genes_in_set_and_de,
    Index total_de_genes,
    Index genes_in_set,
    Index total_genes
);

// TODO: GSEA score
Real gsea_score(
    Array<const Index> ranked_genes,
    Array<const bool> in_gene_set,
    Real& enrichment_score,
    Real& p_value,
    Index n_permutations = 1000
);

// TODO: Over-representation analysis (ORA)
void ora(
    Array<const Index> de_genes,
    const std::vector<std::vector<Index>>& gene_sets,
    std::vector<Real>& p_values,
    std::vector<Real>& odds_ratios
);

// TODO: Pathway Z-score
template <typename T, bool IsCSR>
void pathway_zscore(
    const Sparse<T, IsCSR>& expression,
    const std::vector<std::vector<Index>>& pathways,
    Sparse<Real, !IsCSR>& pathway_scores
);

// TODO: Leading edge genes
void leading_edge(
    Array<const Index> ranked_genes,
    Array<const bool> in_gene_set,
    std::vector<Index>& leading_edge_genes
);

} // namespace scl::kernel::enrichment

