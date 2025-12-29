// =============================================================================
// FILE: scl/binding/c_api/communication/communication.cpp
// BRIEF: C API implementation for communication analysis
// =============================================================================

#include "scl/binding/c_api/communication.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/communication.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

[[nodiscard]] constexpr auto convert_score_method(
    scl_comm_score_method_t method) noexcept -> scl::kernel::communication::ScoreMethod {
    switch (method) {
        case SCL_COMM_SCORE_MEAN_PRODUCT:
            return scl::kernel::communication::ScoreMethod::MeanProduct;
        case SCL_COMM_SCORE_GEOMETRIC_MEAN:
            return scl::kernel::communication::ScoreMethod::GeometricMean;
        case SCL_COMM_SCORE_MIN_MEAN:
            return scl::kernel::communication::ScoreMethod::MinMean;
        case SCL_COMM_SCORE_PRODUCT:
            return scl::kernel::communication::ScoreMethod::Product;
        case SCL_COMM_SCORE_NATMI:
            return scl::kernel::communication::ScoreMethod::Natmi;
        default:
            return scl::kernel::communication::ScoreMethod::MeanProduct;
    }
}

} // anonymous namespace

extern "C" {

// =============================================================================
// L-R Score Matrix
// =============================================================================

SCL_EXPORT scl_error_t scl_comm_lr_score_matrix(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t ligand_gene,
    const scl_index_t receptor_gene,
    const scl_index_t n_cells,
    const scl_index_t n_types,
    scl_real_t* score_matrix,
    const scl_comm_score_method_t method) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(cell_type_labels, "Cell type labels array is null");
    SCL_C_API_CHECK_NULL(score_matrix, "Output score matrix is null");
    SCL_C_API_CHECK(n_cells > 0 && n_types > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        expression->visit([&](auto& m) {
            Array<const Index> labels(cell_type_labels, static_cast<Size>(n_cells));
            Array<Real> scores(
                reinterpret_cast<Real*>(score_matrix), 
                static_cast<Size>(n_types) * static_cast<Size>(n_types)
            );
            
            scl::kernel::communication::lr_score_matrix(
                m, labels, ligand_gene, receptor_gene, n_cells, n_types,
                scores.ptr, convert_score_method(method)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Batch L-R Scores
// =============================================================================

SCL_EXPORT scl_error_t scl_comm_lr_score_batch(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* ligand_genes,
    const scl_index_t* receptor_genes,
    const scl_index_t n_pairs,
    const scl_index_t n_cells,
    const scl_index_t n_types,
    scl_real_t* scores,
    const scl_comm_score_method_t method) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(cell_type_labels, "Cell type labels array is null");
    SCL_C_API_CHECK_NULL(ligand_genes, "Ligand genes array is null");
    SCL_C_API_CHECK_NULL(receptor_genes, "Receptor genes array is null");
    SCL_C_API_CHECK_NULL(scores, "Output scores array is null");
    SCL_C_API_CHECK(n_pairs > 0 && n_cells > 0 && n_types > 0, 
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        expression->visit([&](auto& m) {
            Array<const Index> labels(cell_type_labels, static_cast<Size>(n_cells));
            Array<Real> scores_arr(
                reinterpret_cast<Real*>(scores),
                static_cast<Size>(n_pairs) * static_cast<Size>(n_types) * static_cast<Size>(n_types)
            );
            
            scl::kernel::communication::lr_score_batch(
                m, labels, ligand_genes, receptor_genes, n_pairs, n_cells, n_types,
                scores_arr.ptr, convert_score_method(method)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Permutation Test
// =============================================================================

SCL_EXPORT scl_error_t scl_comm_lr_permutation_test(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t ligand_gene,
    const scl_index_t receptor_gene,
    const scl_index_t sender_type,
    const scl_index_t receiver_type,
    const scl_index_t n_cells,
    const scl_index_t n_permutations,
    scl_real_t* observed_score,
    scl_real_t* p_value,
    const scl_comm_score_method_t method,
    const uint64_t seed) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(cell_type_labels, "Cell type labels array is null");
    SCL_C_API_CHECK_NULL(observed_score, "Output observed score pointer is null");
    SCL_C_API_CHECK_NULL(p_value, "Output p-value pointer is null");
    SCL_C_API_CHECK(n_cells > 0 && n_permutations > 0, 
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        Real obs_score = Real(0);
        Real pval = Real(0);
        
        expression->visit([&](auto& m) {
            Array<const Index> labels(cell_type_labels, static_cast<Size>(n_cells));
            
            scl::kernel::communication::lr_permutation_test(
                m, labels, ligand_gene, receptor_gene, sender_type, receiver_type,
                n_cells, n_permutations, obs_score, pval, 
                convert_score_method(method), seed
            );
        });
        
        *observed_score = static_cast<scl_real_t>(obs_score);
        *p_value = static_cast<scl_real_t>(pval);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Communication Probability
// =============================================================================

SCL_EXPORT scl_error_t scl_comm_probability(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* ligand_genes,
    const scl_index_t* receptor_genes,
    const scl_index_t n_pairs,
    const scl_index_t n_cells,
    const scl_index_t n_types,
    scl_real_t* p_values,
    scl_real_t* scores,
    const scl_index_t n_permutations,
    const scl_comm_score_method_t method,
    const uint64_t seed) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(cell_type_labels, "Cell type labels array is null");
    SCL_C_API_CHECK_NULL(ligand_genes, "Ligand genes array is null");
    SCL_C_API_CHECK_NULL(receptor_genes, "Receptor genes array is null");
    SCL_C_API_CHECK_NULL(p_values, "Output p-values array is null");
    SCL_C_API_CHECK(n_pairs > 0 && n_cells > 0 && n_types > 0 && n_permutations > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        expression->visit([&](auto& m) {
            Array<const Index> labels(cell_type_labels, static_cast<Size>(n_cells));
            const Size result_size = static_cast<Size>(n_pairs) * static_cast<Size>(n_types) * static_cast<Size>(n_types);
            
            Array<Real> pvals(reinterpret_cast<Real*>(p_values), result_size);
            Array<Real> scores_arr;
            if (scores) {
                scores_arr = Array<Real>(reinterpret_cast<Real*>(scores), result_size);
            }
            
            scl::kernel::communication::communication_probability(
                m, labels, ligand_genes, receptor_genes, n_pairs, n_cells, n_types,
                pvals.ptr, scores ? scores_arr.ptr : nullptr, n_permutations,
                convert_score_method(method), seed
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Filter Significant Interactions
// =============================================================================

SCL_EXPORT scl_error_t scl_comm_filter_significant(
    const scl_real_t* p_values,
    const scl_index_t n_pairs,
    const scl_index_t n_types,
    const scl_real_t p_threshold,
    scl_index_t* pair_indices,
    scl_index_t* sender_types,
    scl_index_t* receiver_types,
    scl_real_t* filtered_pvalues,
    const scl_index_t max_results,
    scl_index_t* n_results) {
    
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK_NULL(pair_indices, "Output pair indices array is null");
    SCL_C_API_CHECK_NULL(sender_types, "Output sender types array is null");
    SCL_C_API_CHECK_NULL(receiver_types, "Output receiver types array is null");
    SCL_C_API_CHECK_NULL(filtered_pvalues, "Output filtered p-values array is null");
    SCL_C_API_CHECK_NULL(n_results, "Output n_results pointer is null");
    SCL_C_API_CHECK(n_pairs > 0 && n_types > 0 && max_results > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Size result_size = static_cast<Size>(n_pairs) * static_cast<Size>(n_types) * static_cast<Size>(n_types);
        
        Array<const Real> pvals(reinterpret_cast<const Real*>(p_values), result_size);
        Array<Index> pairs(pair_indices, static_cast<Size>(max_results));
        Array<Index> senders(sender_types, static_cast<Size>(max_results));
        Array<Index> receivers(receiver_types, static_cast<Size>(max_results));
        Array<Real> filtered(reinterpret_cast<Real*>(filtered_pvalues), static_cast<Size>(max_results));
        
        Index count = scl::kernel::communication::filter_significant(
            pvals.ptr, n_pairs, n_types, static_cast<Real>(p_threshold),
            pairs.ptr, senders.ptr, receivers.ptr, filtered.ptr, max_results
        );
        
        *n_results = count;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Aggregate to Network
// =============================================================================

SCL_EXPORT scl_error_t scl_comm_aggregate_to_network(
    const scl_real_t* scores,
    const scl_real_t* p_values,
    const scl_index_t n_pairs,
    const scl_index_t n_types,
    const scl_real_t p_threshold,
    scl_real_t* network_weights,
    scl_index_t* network_counts) {
    
    SCL_C_API_CHECK_NULL(scores, "Scores array is null");
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK_NULL(network_weights, "Output network weights array is null");
    SCL_C_API_CHECK_NULL(network_counts, "Output network counts array is null");
    SCL_C_API_CHECK(n_pairs > 0 && n_types > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Size input_size = static_cast<Size>(n_pairs) * static_cast<Size>(n_types) * static_cast<Size>(n_types);
        const Size output_size = static_cast<Size>(n_types) * static_cast<Size>(n_types);
        
        Array<const Real> scores_arr(reinterpret_cast<const Real*>(scores), input_size);
        Array<const Real> pvals(reinterpret_cast<const Real*>(p_values), input_size);
        Array<Real> weights(reinterpret_cast<Real*>(network_weights), output_size);
        Array<Index> counts(network_counts, output_size);
        
        scl::kernel::communication::aggregate_to_network(
            scores_arr.ptr, pvals.ptr, n_pairs, n_types, static_cast<Real>(p_threshold),
            weights.ptr, counts.ptr
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Sender/Receiver Scores
// =============================================================================

SCL_EXPORT scl_error_t scl_comm_sender_score(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* ligand_genes,
    const scl_index_t n_ligands,
    const scl_index_t n_cells,
    const scl_index_t n_types,
    scl_real_t* scores) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(cell_type_labels, "Cell type labels array is null");
    SCL_C_API_CHECK_NULL(ligand_genes, "Ligand genes array is null");
    SCL_C_API_CHECK_NULL(scores, "Output scores array is null");
    SCL_C_API_CHECK(n_ligands > 0 && n_cells > 0 && n_types > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        expression->visit([&](auto& m) {
            Array<const Index> labels(cell_type_labels, static_cast<Size>(n_cells));
            Array<Real> scores_arr(reinterpret_cast<Real*>(scores), static_cast<Size>(n_types));
            
            scl::kernel::communication::sender_score(
                m, labels, ligand_genes, n_ligands, n_cells, n_types, scores_arr.ptr
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_comm_receiver_score(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* receptor_genes,
    const scl_index_t n_receptors,
    const scl_index_t n_cells,
    const scl_index_t n_types,
    scl_real_t* scores) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(cell_type_labels, "Cell type labels array is null");
    SCL_C_API_CHECK_NULL(receptor_genes, "Receptor genes array is null");
    SCL_C_API_CHECK_NULL(scores, "Output scores array is null");
    SCL_C_API_CHECK(n_receptors > 0 && n_cells > 0 && n_types > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        expression->visit([&](auto& m) {
            Array<const Index> labels(cell_type_labels, static_cast<Size>(n_cells));
            Array<Real> scores_arr(reinterpret_cast<Real*>(scores), static_cast<Size>(n_types));
            
            scl::kernel::communication::receiver_score(
                m, labels, receptor_genes, n_receptors, n_cells, n_types, scores_arr.ptr
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Network Centrality
// =============================================================================

SCL_EXPORT scl_error_t scl_comm_network_centrality(
    const scl_real_t* network_weights,
    const scl_index_t n_types,
    scl_real_t* in_degree,
    scl_real_t* out_degree,
    scl_real_t* betweenness) {
    
    SCL_C_API_CHECK_NULL(network_weights, "Network weights array is null");
    SCL_C_API_CHECK_NULL(in_degree, "Output in-degree array is null");
    SCL_C_API_CHECK_NULL(out_degree, "Output out-degree array is null");
    SCL_C_API_CHECK(n_types > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of types must be positive");
    
    SCL_C_API_TRY
        const Size network_size = static_cast<Size>(n_types) * static_cast<Size>(n_types);
        
        Array<const Real> weights(reinterpret_cast<const Real*>(network_weights), network_size);
        Array<Real> in_deg(reinterpret_cast<Real*>(in_degree), static_cast<Size>(n_types));
        Array<Real> out_deg(reinterpret_cast<Real*>(out_degree), static_cast<Size>(n_types));
        Array<Real> betw;
        if (betweenness) {
            betw = Array<Real>(reinterpret_cast<Real*>(betweenness), static_cast<Size>(n_types));
        }
        
        scl::kernel::communication::network_centrality(
            weights.ptr, n_types, in_deg.ptr, out_deg.ptr, betweenness ? betw.ptr : nullptr
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Spatial Communication Score
// =============================================================================

SCL_EXPORT scl_error_t scl_comm_spatial_score(
    scl_sparse_t expression,
    scl_sparse_t spatial_graph,
    const scl_index_t ligand_gene,
    const scl_index_t receptor_gene,
    const scl_index_t n_cells,
    scl_real_t* cell_scores) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(spatial_graph, "Spatial graph is null");
    SCL_C_API_CHECK_NULL(cell_scores, "Output cell scores array is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    
    SCL_C_API_TRY
        Array<Real> scores(reinterpret_cast<Real*>(cell_scores), static_cast<Size>(n_cells));
        
        expression->visit([&](auto& expr) {
            spatial_graph->visit([&](auto& graph) {
                scl::kernel::communication::spatial_communication_score(
                    expr, graph, ligand_gene, receptor_gene, n_cells, scores.ptr
                );
            });
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Expression Specificity
// =============================================================================

SCL_EXPORT scl_error_t scl_comm_expression_specificity(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t gene,
    const scl_index_t n_cells,
    const scl_index_t n_types,
    scl_real_t* specificity) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(cell_type_labels, "Cell type labels array is null");
    SCL_C_API_CHECK_NULL(specificity, "Output specificity array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_types > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        expression->visit([&](auto& m) {
            Array<const Index> labels(cell_type_labels, static_cast<Size>(n_cells));
            Array<Real> spec(reinterpret_cast<Real*>(specificity), static_cast<Size>(n_types));
            
            scl::kernel::communication::expression_specificity(
                m, labels, gene, n_cells, n_types, spec.ptr
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// NATMI Edge Weight
// =============================================================================

SCL_EXPORT scl_error_t scl_comm_natmi_edge_weight(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t ligand_gene,
    const scl_index_t receptor_gene,
    const scl_index_t n_cells,
    const scl_index_t n_types,
    scl_real_t* edge_weights) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(cell_type_labels, "Cell type labels array is null");
    SCL_C_API_CHECK_NULL(edge_weights, "Output edge weights array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_types > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        expression->visit([&](auto& m) {
            Array<const Index> labels(cell_type_labels, static_cast<Size>(n_cells));
            const Size weights_size = static_cast<Size>(n_types) * static_cast<Size>(n_types);
            Array<Real> weights(reinterpret_cast<Real*>(edge_weights), weights_size);
            
            scl::kernel::communication::natmi_edge_weight(
                m, labels, ligand_gene, receptor_gene, n_cells, n_types, weights.ptr
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
