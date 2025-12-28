// =============================================================================
// FILE: scl/binding/c_api/communication/communication.cpp
// BRIEF: C API implementation for communication analysis
// =============================================================================

#include "scl/binding/c_api/communication.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/communication.hpp"
#include "scl/core/type.hpp"

#include <exception>

extern "C" {

namespace {
    scl::kernel::communication::ScoreMethod convert_score_method(scl_comm_score_method_t method) {
        switch (method) {
            case SCL_COMM_SCORE_MEAN_PRODUCT: return scl::kernel::communication::ScoreMethod::MeanProduct;
            case SCL_COMM_SCORE_GEOMETRIC_MEAN: return scl::kernel::communication::ScoreMethod::GeometricMean;
            case SCL_COMM_SCORE_MIN_MEAN: return scl::kernel::communication::ScoreMethod::MinMean;
            case SCL_COMM_SCORE_PRODUCT: return scl::kernel::communication::ScoreMethod::Product;
            case SCL_COMM_SCORE_NATMI: return scl::kernel::communication::ScoreMethod::Natmi;
            default: return scl::kernel::communication::ScoreMethod::MeanProduct;
        }
    }
}

scl_error_t scl_comm_lr_score_matrix(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    scl_index_t ligand_gene,
    scl_index_t receptor_gene,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* score_matrix,
    scl_comm_score_method_t method
) {
    if (!expression || !cell_type_labels || !score_matrix) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(score_matrix), 
                                     static_cast<scl::Size>(n_types) * n_types);

        sparse->visit([&](auto& m) {
            scl::kernel::communication::lr_score_matrix(
                m, labels, ligand_gene, receptor_gene, n_cells, n_types,
                scores.ptr, convert_score_method(method)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comm_lr_score_batch(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* ligand_genes,
    const scl_index_t* receptor_genes,
    scl_index_t n_pairs,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* scores,
    scl_comm_score_method_t method
) {
    if (!expression || !cell_type_labels || !ligand_genes || !receptor_genes || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> scores_arr(reinterpret_cast<scl::Real*>(scores),
                                         static_cast<scl::Size>(n_pairs) * n_types * n_types);

        sparse->visit([&](auto& m) {
            scl::kernel::communication::lr_score_batch(
                m, labels, ligand_genes, receptor_genes, n_pairs, n_cells, n_types,
                scores_arr.ptr, convert_score_method(method)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comm_lr_permutation_test(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    scl_index_t ligand_gene,
    scl_index_t receptor_gene,
    scl_index_t sender_type,
    scl_index_t receiver_type,
    scl_index_t n_cells,
    scl_index_t n_permutations,
    scl_real_t* observed_score,
    scl_real_t* p_value,
    scl_comm_score_method_t method,
    uint64_t seed
) {
    if (!expression || !cell_type_labels || !observed_score || !p_value) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(n_cells));
        scl::Real obs_score, pval;

        sparse->visit([&](auto& m) {
            scl::kernel::communication::lr_permutation_test(
                m, labels, ligand_gene, receptor_gene, sender_type, receiver_type,
                n_cells, n_permutations, obs_score, pval, convert_score_method(method), seed
            );
        });

        *observed_score = static_cast<scl_real_t>(obs_score);
        *p_value = static_cast<scl_real_t>(pval);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comm_probability(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* ligand_genes,
    const scl_index_t* receptor_genes,
    scl_index_t n_pairs,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* p_values,
    scl_real_t* scores,
    scl_index_t n_permutations,
    scl_comm_score_method_t method,
    uint64_t seed
) {
    if (!expression || !cell_type_labels || !ligand_genes || !receptor_genes || !p_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> pvals(reinterpret_cast<scl::Real*>(p_values),
                                     static_cast<scl::Size>(n_pairs) * n_types * n_types);
        scl::Array<scl::Real> scores_arr;
        if (scores) {
            scores_arr = scl::Array<scl::Real>(reinterpret_cast<scl::Real*>(scores),
                                               static_cast<scl::Size>(n_pairs) * n_types * n_types);
        }

        sparse->visit([&](auto& m) {
            scl::kernel::communication::communication_probability(
                m, labels, ligand_genes, receptor_genes, n_pairs, n_cells, n_types,
                pvals.ptr, scores ? scores_arr.ptr : nullptr, n_permutations,
                convert_score_method(method), seed
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comm_filter_significant(
    const scl_real_t* p_values,
    scl_index_t n_pairs,
    scl_index_t n_types,
    scl_real_t p_threshold,
    scl_index_t* pair_indices,
    scl_index_t* sender_types,
    scl_index_t* receiver_types,
    scl_real_t* filtered_pvalues,
    scl_index_t max_results,
    scl_index_t* n_results
) {
    if (!p_values || !pair_indices || !sender_types || !receiver_types || !filtered_pvalues || !n_results) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> pvals(reinterpret_cast<const scl::Real*>(p_values),
                                          static_cast<scl::Size>(n_pairs) * n_types * n_types);
        scl::Array<scl::Index> pairs(pair_indices, static_cast<scl::Size>(max_results));
        scl::Array<scl::Index> senders(sender_types, static_cast<scl::Size>(max_results));
        scl::Array<scl::Index> receivers(receiver_types, static_cast<scl::Size>(max_results));
        scl::Array<scl::Real> filtered(reinterpret_cast<scl::Real*>(filtered_pvalues),
                                       static_cast<scl::Size>(max_results));

        scl::Index count = scl::kernel::communication::filter_significant(
            pvals.ptr, n_pairs, n_types, static_cast<scl::Real>(p_threshold),
            pairs.ptr, senders.ptr, receivers.ptr, filtered.ptr, max_results
        );

        *n_results = count;
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comm_aggregate_to_network(
    const scl_real_t* scores,
    const scl_real_t* p_values,
    scl_index_t n_pairs,
    scl_index_t n_types,
    scl_real_t p_threshold,
    scl_real_t* network_weights,
    scl_index_t* network_counts
) {
    if (!scores || !p_values || !network_weights || !network_counts) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> scores_arr(reinterpret_cast<const scl::Real*>(scores),
                                                static_cast<scl::Size>(n_pairs) * n_types * n_types);
        scl::Array<const scl::Real> pvals(reinterpret_cast<const scl::Real*>(p_values),
                                          static_cast<scl::Size>(n_pairs) * n_types * n_types);
        scl::Array<scl::Real> weights(reinterpret_cast<scl::Real*>(network_weights),
                                      static_cast<scl::Size>(n_types) * n_types);
        scl::Array<scl::Index> counts(network_counts, static_cast<scl::Size>(n_types) * n_types);

        scl::kernel::communication::aggregate_to_network(
            scores_arr.ptr, pvals.ptr, n_pairs, n_types, static_cast<scl::Real>(p_threshold),
            weights.ptr, counts.ptr
        );

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comm_sender_score(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* ligand_genes,
    scl_index_t n_ligands,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* scores
) {
    if (!expression || !cell_type_labels || !ligand_genes || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> scores_arr(reinterpret_cast<scl::Real*>(scores), static_cast<scl::Size>(n_types));

        sparse->visit([&](auto& m) {
            scl::kernel::communication::sender_score(
                m, labels, ligand_genes, n_ligands, n_cells, n_types, scores_arr.ptr
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comm_receiver_score(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* receptor_genes,
    scl_index_t n_receptors,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* scores
) {
    if (!expression || !cell_type_labels || !receptor_genes || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> scores_arr(reinterpret_cast<scl::Real*>(scores), static_cast<scl::Size>(n_types));

        sparse->visit([&](auto& m) {
            scl::kernel::communication::receiver_score(
                m, labels, receptor_genes, n_receptors, n_cells, n_types, scores_arr.ptr
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comm_network_centrality(
    const scl_real_t* network_weights,
    scl_index_t n_types,
    scl_real_t* in_degree,
    scl_real_t* out_degree,
    scl_real_t* betweenness
) {
    if (!network_weights || !in_degree || !out_degree) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> weights(reinterpret_cast<const scl::Real*>(network_weights),
                                           static_cast<scl::Size>(n_types) * n_types);
        scl::Array<scl::Real> in_deg(reinterpret_cast<scl::Real*>(in_degree), static_cast<scl::Size>(n_types));
        scl::Array<scl::Real> out_deg(reinterpret_cast<scl::Real*>(out_degree), static_cast<scl::Size>(n_types));
        scl::Array<scl::Real> betw;
        if (betweenness) {
            betw = scl::Array<scl::Real>(reinterpret_cast<scl::Real*>(betweenness), static_cast<scl::Size>(n_types));
        }

        scl::kernel::communication::network_centrality(
            weights.ptr, n_types, in_deg.ptr, out_deg.ptr, betweenness ? betw.ptr : nullptr
        );

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comm_spatial_score(
    scl_sparse_t expression,
    scl_sparse_t spatial_graph,
    scl_index_t ligand_gene,
    scl_index_t receptor_gene,
    scl_index_t n_cells,
    scl_real_t* cell_scores
) {
    if (!expression || !spatial_graph || !cell_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* expr_sparse = static_cast<scl_sparse_matrix*>(expression);
        auto* graph_sparse = static_cast<scl_sparse_matrix*>(spatial_graph);
        scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(cell_scores), static_cast<scl::Size>(n_cells));

        expr_sparse->visit([&](auto& expr) {
            graph_sparse->visit([&](auto& graph) {
                scl::kernel::communication::spatial_communication_score(
                    expr, graph, ligand_gene, receptor_gene, n_cells, scores.ptr
                );
            });
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comm_expression_specificity(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    scl_index_t gene,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* specificity
) {
    if (!expression || !cell_type_labels || !specificity) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> spec(reinterpret_cast<scl::Real*>(specificity), static_cast<scl::Size>(n_types));

        sparse->visit([&](auto& m) {
            scl::kernel::communication::expression_specificity(
                m, labels, gene, n_cells, n_types, spec.ptr
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comm_natmi_edge_weight(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    scl_index_t ligand_gene,
    scl_index_t receptor_gene,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* edge_weights
) {
    if (!expression || !cell_type_labels || !edge_weights) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> weights(reinterpret_cast<scl::Real*>(edge_weights),
                                      static_cast<scl::Size>(n_types) * n_types);

        sparse->visit([&](auto& m) {
            scl::kernel::communication::natmi_edge_weight(
                m, labels, ligand_gene, receptor_gene, n_cells, n_types, weights.ptr
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
