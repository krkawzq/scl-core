#pragma once

// =============================================================================
// FILE: scl/binding/c_api/state/state.h
// BRIEF: C API for cell state scoring
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Stemness Score
// =============================================================================

scl_error_t scl_state_stemness_score(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    const scl_index_t* stemness_genes,     // [n_stemness_genes]
    scl_size_t n_stemness_genes,
    scl_real_t* scores                     // Output [n_cells]
);

// =============================================================================
// Differentiation Potential (CytoTRACE-style)
// =============================================================================

scl_error_t scl_state_differentiation_potential(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    scl_real_t* potency_scores              // Output [n_cells]
);

// =============================================================================
// Proliferation Score
// =============================================================================

scl_error_t scl_state_proliferation_score(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    const scl_index_t* proliferation_genes, // [n_proliferation_genes]
    scl_size_t n_proliferation_genes,
    scl_real_t* scores                      // Output [n_cells]
);

// =============================================================================
// Stress Score
// =============================================================================

scl_error_t scl_state_stress_score(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    const scl_index_t* stress_genes,        // [n_stress_genes]
    scl_size_t n_stress_genes,
    scl_real_t* scores                      // Output [n_cells]
);

// =============================================================================
// State Entropy (Plasticity)
// =============================================================================

scl_error_t scl_state_entropy(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    scl_real_t* entropy_scores              // Output [n_cells]
);

// =============================================================================
// Cell Cycle Score (G1/S/G2M)
// =============================================================================

scl_error_t scl_state_cell_cycle_score(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    const scl_index_t* s_genes,             // [n_s_genes]
    scl_size_t n_s_genes,
    const scl_index_t* g2m_genes,           // [n_g2m_genes]
    scl_size_t n_g2m_genes,
    scl_real_t* s_scores,                   // Output [n_cells]
    scl_real_t* g2m_scores,                 // Output [n_cells]
    scl_index_t* phase_labels               // Output [n_cells]: 0=G1, 1=S, 2=G2M
);

// =============================================================================
// Quiescence Score
// =============================================================================

scl_error_t scl_state_quiescence_score(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    const scl_index_t* quiescence_genes,    // [n_quiescence_genes]
    scl_size_t n_quiescence_genes,
    const scl_index_t* proliferation_genes, // [n_proliferation_genes]
    scl_size_t n_proliferation_genes,
    scl_real_t* scores                      // Output [n_cells]
);

// =============================================================================
// Metabolic Activity Score
// =============================================================================

scl_error_t scl_state_metabolic_score(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    const scl_index_t* glycolysis_genes,    // [n_glycolysis_genes]
    scl_size_t n_glycolysis_genes,
    const scl_index_t* oxphos_genes,        // [n_oxphos_genes]
    scl_size_t n_oxphos_genes,
    scl_real_t* glycolysis_scores,           // Output [n_cells]
    scl_real_t* oxphos_scores               // Output [n_cells]
);

// =============================================================================
// Apoptosis Score
// =============================================================================

scl_error_t scl_state_apoptosis_score(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    const scl_index_t* apoptosis_genes,     // [n_apoptosis_genes]
    scl_size_t n_apoptosis_genes,
    scl_real_t* scores                       // Output [n_cells]
);

// =============================================================================
// Gene Signature Score (generalized)
// =============================================================================

scl_error_t scl_state_signature_score(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    const scl_index_t* gene_indices,        // [n_signature]
    scl_size_t n_signature,
    const scl_real_t* gene_weights,         // [n_signature]
    scl_real_t* scores                       // Output [n_cells]
);

// =============================================================================
// Multi-signature Score Matrix
// =============================================================================

scl_error_t scl_state_multi_signature_score(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    const scl_index_t* signature_gene_indices, // Flat array of gene indices
    const scl_size_t* signature_offsets,      // [n_signatures + 1] Start offset for each signature
    scl_size_t n_signatures,
    scl_real_t* score_matrix                // Output [n_cells * n_signatures]
);

// =============================================================================
// Transcriptional Diversity
// =============================================================================

scl_error_t scl_state_transcriptional_diversity(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    scl_real_t* diversity_scores             // Output [n_cells]
);

// =============================================================================
// Expression Complexity
// =============================================================================

scl_error_t scl_state_expression_complexity(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    scl_real_t expression_threshold,
    scl_real_t* complexity_scores            // Output [n_cells]
);

// =============================================================================
// Combined State Score
// =============================================================================

scl_error_t scl_state_combined_score(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    const scl_index_t* const* gene_sets,    // Array of gene set pointers
    const scl_size_t* gene_set_sizes,       // [n_gene_sets] Size of each gene set
    const scl_real_t* weights,              // [n_gene_sets] Weight for each gene set
    scl_size_t n_gene_sets,
    scl_real_t* combined_scores              // Output [n_cells]
);

#ifdef __cplusplus
}
#endif
