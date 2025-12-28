// =============================================================================
// FILE: scl/binding/c_api/state/state.cpp
// BRIEF: C API implementation for cell state scoring
// =============================================================================

#include "scl/binding/c_api/state.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/state.hpp"
#include "scl/core/type.hpp"

extern "C" {

scl_error_t scl_state_stemness_score(
    scl_sparse_t expression,
    const scl_index_t* stemness_genes,
    scl_size_t n_stemness_genes,
    scl_real_t* scores
) {
    if (!expression || !stemness_genes || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Index> genes(stemness_genes, n_stemness_genes);
            scl::Array<scl::Real> score_arr(reinterpret_cast<scl::Real*>(scores),
                                           static_cast<scl::Size>(n_cells));
            scl::kernel::state::stemness_score(m, genes, score_arr);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_state_differentiation_potential(
    scl_sparse_t expression,
    scl_real_t* potency_scores
) {
    if (!expression || !potency_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(potency_scores),
                                        static_cast<scl::Size>(n_cells));
            scl::kernel::state::differentiation_potential(m, scores);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_state_proliferation_score(
    scl_sparse_t expression,
    const scl_index_t* proliferation_genes,
    scl_size_t n_proliferation_genes,
    scl_real_t* scores
) {
    if (!expression || !proliferation_genes || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Index> genes(proliferation_genes, n_proliferation_genes);
            scl::Array<scl::Real> score_arr(reinterpret_cast<scl::Real*>(scores),
                                           static_cast<scl::Size>(n_cells));
            scl::kernel::state::proliferation_score(m, genes, score_arr);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_state_stress_score(
    scl_sparse_t expression,
    const scl_index_t* stress_genes,
    scl_size_t n_stress_genes,
    scl_real_t* scores
) {
    if (!expression || !stress_genes || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Index> genes(stress_genes, n_stress_genes);
            scl::Array<scl::Real> score_arr(reinterpret_cast<scl::Real*>(scores),
                                           static_cast<scl::Size>(n_cells));
            scl::kernel::state::stress_score(m, genes, score_arr);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_state_entropy(
    scl_sparse_t expression,
    scl_real_t* entropy_scores
) {
    if (!expression || !entropy_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(entropy_scores),
                                        static_cast<scl::Size>(n_cells));
            scl::kernel::state::state_entropy(m, scores);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_state_cell_cycle_score(
    scl_sparse_t expression,
    const scl_index_t* s_genes,
    scl_size_t n_s_genes,
    const scl_index_t* g2m_genes,
    scl_size_t n_g2m_genes,
    scl_real_t* s_scores,
    scl_real_t* g2m_scores,
    scl_index_t* phase_labels
) {
    if (!expression || !s_genes || !g2m_genes || !s_scores || !g2m_scores || !phase_labels) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Index> s_arr(s_genes, n_s_genes);
            scl::Array<const scl::Index> g2m_arr(g2m_genes, n_g2m_genes);
            scl::Array<scl::Real> s_score_arr(reinterpret_cast<scl::Real*>(s_scores),
                                             static_cast<scl::Size>(n_cells));
            scl::Array<scl::Real> g2m_score_arr(reinterpret_cast<scl::Real*>(g2m_scores),
                                                static_cast<scl::Size>(n_cells));
            scl::Array<scl::Index> phase_arr(phase_labels, static_cast<scl::Size>(n_cells));

            scl::kernel::state::cell_cycle_score(
                m, s_arr, g2m_arr, s_score_arr, g2m_score_arr, phase_arr
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_state_quiescence_score(
    scl_sparse_t expression,
    const scl_index_t* quiescence_genes,
    scl_size_t n_quiescence_genes,
    const scl_index_t* proliferation_genes,
    scl_size_t n_proliferation_genes,
    scl_real_t* scores
) {
    if (!expression || !quiescence_genes || !proliferation_genes || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Index> q_genes(quiescence_genes, n_quiescence_genes);
            scl::Array<const scl::Index> p_genes(proliferation_genes, n_proliferation_genes);
            scl::Array<scl::Real> score_arr(reinterpret_cast<scl::Real*>(scores),
                                           static_cast<scl::Size>(n_cells));
            scl::kernel::state::quiescence_score(m, q_genes, p_genes, score_arr);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_state_metabolic_score(
    scl_sparse_t expression,
    const scl_index_t* glycolysis_genes,
    scl_size_t n_glycolysis_genes,
    const scl_index_t* oxphos_genes,
    scl_size_t n_oxphos_genes,
    scl_real_t* glycolysis_scores,
    scl_real_t* oxphos_scores
) {
    if (!expression || !glycolysis_genes || !oxphos_genes || !glycolysis_scores || !oxphos_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Index> gly_genes(glycolysis_genes, n_glycolysis_genes);
            scl::Array<const scl::Index> ox_genes(oxphos_genes, n_oxphos_genes);
            scl::Array<scl::Real> gly_scores(reinterpret_cast<scl::Real*>(glycolysis_scores),
                                            static_cast<scl::Size>(n_cells));
            scl::Array<scl::Real> ox_scores(reinterpret_cast<scl::Real*>(oxphos_scores),
                                           static_cast<scl::Size>(n_cells));
            scl::kernel::state::metabolic_score(m, gly_genes, ox_genes, gly_scores, ox_scores);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_state_apoptosis_score(
    scl_sparse_t expression,
    const scl_index_t* apoptosis_genes,
    scl_size_t n_apoptosis_genes,
    scl_real_t* scores
) {
    if (!expression || !apoptosis_genes || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Index> genes(apoptosis_genes, n_apoptosis_genes);
            scl::Array<scl::Real> score_arr(reinterpret_cast<scl::Real*>(scores),
                                           static_cast<scl::Size>(n_cells));
            scl::kernel::state::apoptosis_score(m, genes, score_arr);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_state_signature_score(
    scl_sparse_t expression,
    const scl_index_t* gene_indices,
    scl_size_t n_signature,
    const scl_real_t* gene_weights,
    scl_real_t* scores
) {
    if (!expression || !gene_indices || !gene_weights || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Index> genes(gene_indices, n_signature);
            scl::Array<const scl::Real> weights(reinterpret_cast<const scl::Real*>(gene_weights),
                                                n_signature);
            scl::Array<scl::Real> score_arr(reinterpret_cast<scl::Real*>(scores),
                                           static_cast<scl::Size>(n_cells));
            scl::kernel::state::signature_score(m, genes, weights, score_arr);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_state_multi_signature_score(
    scl_sparse_t expression,
    const scl_index_t* signature_gene_indices,
    const scl_size_t* signature_offsets,
    scl_size_t n_signatures,
    scl_real_t* score_matrix
) {
    if (!expression || !signature_gene_indices || !signature_offsets || !score_matrix) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);

        sparse->visit([&](auto& m) {
            scl::kernel::state::multi_signature_score(
                m, signature_gene_indices, signature_offsets, n_signatures,
                reinterpret_cast<scl::Real*>(score_matrix)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_state_transcriptional_diversity(
    scl_sparse_t expression,
    scl_real_t* diversity_scores
) {
    if (!expression || !diversity_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(diversity_scores),
                                        static_cast<scl::Size>(n_cells));
            scl::kernel::state::transcriptional_diversity(m, scores);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_state_expression_complexity(
    scl_sparse_t expression,
    scl_real_t expression_threshold,
    scl_real_t* complexity_scores
) {
    if (!expression || !complexity_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(complexity_scores),
                                        static_cast<scl::Size>(n_cells));
            scl::kernel::state::expression_complexity(
                m, static_cast<scl::Real>(expression_threshold), scores
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_state_combined_score(
    scl_sparse_t expression,
    const scl_index_t* const* gene_sets,
    const scl_size_t* gene_set_sizes,
    const scl_real_t* weights,
    scl_size_t n_gene_sets,
    scl_real_t* combined_scores
) {
    if (!expression || !gene_sets || !gene_set_sizes || !weights || !combined_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(combined_scores),
                                        static_cast<scl::Size>(n_cells));
            scl::kernel::state::combined_state_score(
                m, gene_sets, gene_set_sizes,
                reinterpret_cast<const scl::Real*>(weights),
                n_gene_sets, scores
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
