// =============================================================================
// FILE: scl/binding/c_api/annotation.cpp
// BRIEF: C API implementation for cell type annotation
// =============================================================================

#include "scl/binding/c_api/annotation.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/annotation.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Reference Mapping
// =============================================================================

SCL_EXPORT scl_error_t scl_annotation_reference_mapping(
    scl_sparse_t query_expression,
    scl_sparse_t reference_expression,
    const scl_index_t* reference_labels,
    const scl_size_t n_ref,
    scl_sparse_t query_to_ref_neighbors,
    const scl_index_t n_query,
    const scl_index_t n_types,
    scl_index_t* query_labels,
    scl_real_t* confidence_scores) {
    
    SCL_C_API_CHECK_NULL(query_expression, "Query expression matrix is null");
    SCL_C_API_CHECK_NULL(reference_expression, "Reference expression matrix is null");
    SCL_C_API_CHECK_NULL(reference_labels, "Reference labels array is null");
    SCL_C_API_CHECK_NULL(query_to_ref_neighbors, "Neighbor graph is null");
    SCL_C_API_CHECK_NULL(query_labels, "Output query labels array is null");
    SCL_C_API_CHECK_NULL(confidence_scores, "Output confidence scores array is null");
    
    SCL_C_API_TRY
        query_expression->visit([&](auto& query) {
            using QueryType = std::remove_reference_t<decltype(query)>;
            using T = typename QueryType::ValueType;
            constexpr bool IsCSR_Query = QueryType::is_csr;
            
            reference_expression->visit([&](auto& ref) {
                using RefType = std::remove_reference_t<decltype(ref)>;
                constexpr bool IsCSR_Ref = RefType::is_csr;
                
                query_to_ref_neighbors->visit([&](auto& neighbors_real) {
                    using NeighborType = std::remove_reference_t<decltype(neighbors_real)>;
                    constexpr bool IsCSR_Neighbors = NeighborType::is_csr;
                    
                    // PERFORMANCE: reinterpret Real sparse as Index sparse (same layout)
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                    const auto& neighbors = reinterpret_cast<
                        const Sparse<Index, IsCSR_Neighbors>&>(neighbors_real);
                    
                    scl::kernel::annotation::reference_mapping<T, IsCSR_Query, IsCSR_Ref, IsCSR_Neighbors>(
                        query, ref,
                        Array<const Index>(
                            reinterpret_cast<const Index*>(reference_labels),
                            n_ref
                        ),
                        neighbors,
                        n_query,
                        static_cast<Index>(n_ref),
                        n_types,
                        Array<Index>(
                            reinterpret_cast<Index*>(query_labels),
                            static_cast<Size>(n_query)
                        ),
                        Array<Real>(
                            reinterpret_cast<Real*>(confidence_scores),
                            static_cast<Size>(n_query)
                        )
                    );
                });
            });
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Correlation Assignment
// =============================================================================

SCL_EXPORT scl_error_t scl_annotation_correlation_assignment(
    scl_sparse_t query_expression,
    scl_sparse_t reference_profiles,
    const scl_index_t n_query,
    const scl_index_t n_types,
    const scl_index_t n_genes,
    scl_index_t* assigned_labels,
    scl_real_t* correlation_scores,
    scl_real_t* all_correlations) {
    
    SCL_C_API_CHECK_NULL(query_expression, "Query expression matrix is null");
    SCL_C_API_CHECK_NULL(reference_profiles, "Reference profiles matrix is null");
    SCL_C_API_CHECK_NULL(assigned_labels, "Output labels array is null");
    SCL_C_API_CHECK_NULL(correlation_scores, "Output correlation scores array is null");
    
    SCL_C_API_TRY
        query_expression->visit([&](auto& query) {
            reference_profiles->visit([&](auto& profiles) {
                const Size all_corr_size = static_cast<Size>(n_query) * static_cast<Size>(n_types);
                
                scl::kernel::annotation::correlation_assignment(
                    query, profiles,
                    n_query,
                    n_types,
                    n_genes,
                    Array<Index>(
                        reinterpret_cast<Index*>(assigned_labels),
                        static_cast<Size>(n_query)
                    ),
                    Array<Real>(
                        reinterpret_cast<Real*>(correlation_scores),
                        static_cast<Size>(n_query)
                    ),
                    all_correlations ? Array<Real>(
                        reinterpret_cast<Real*>(all_correlations),
                        all_corr_size
                    ) : Array<Real>(nullptr, 0)
                );
            });
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Build Reference Profiles
// =============================================================================

SCL_EXPORT scl_error_t scl_annotation_build_reference_profiles(
    scl_sparse_t expression,
    const scl_index_t* labels,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_index_t n_types,
    scl_real_t* profiles) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(labels, "Labels array is null");
    SCL_C_API_CHECK_NULL(profiles, "Output profiles array is null");
    
    SCL_C_API_TRY
        expression->visit([&](auto& expr) {
            scl::kernel::annotation::build_reference_profiles(
                expr,
                Array<const Index>(
                    reinterpret_cast<const Index*>(labels),
                    static_cast<Size>(n_cells)
                ),
                n_cells,
                n_genes,
                n_types,
                reinterpret_cast<Real*>(profiles)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Marker Gene Score
// =============================================================================

SCL_EXPORT scl_error_t scl_annotation_marker_gene_score(
    scl_sparse_t expression,
    const scl_index_t* const* marker_genes,
    const scl_index_t* marker_counts,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_index_t n_types,
    scl_real_t* scores,
    const scl_bool_t normalize) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(marker_genes, "Marker genes array is null");
    SCL_C_API_CHECK_NULL(marker_counts, "Marker counts array is null");
    SCL_C_API_CHECK_NULL(scores, "Output scores array is null");
    
    SCL_C_API_TRY
        // PERFORMANCE: C array cast - safe due to compatible layout
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        const auto* const* marker_arrays = 
            reinterpret_cast<const Index* const*>(marker_genes);
        
        expression->visit([&](auto& expr) {
            scl::kernel::annotation::marker_gene_score(
                expr,
                marker_arrays,
                reinterpret_cast<const Index*>(marker_counts),
                n_cells,
                n_genes,
                n_types,
                reinterpret_cast<Real*>(scores),
                normalize != SCL_FALSE
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_annotation_assign_from_marker_scores(
    const scl_real_t* scores,
    const scl_index_t n_cells,
    const scl_index_t n_types,
    scl_index_t* labels,
    scl_real_t* confidence) {
    
    SCL_C_API_CHECK_NULL(scores, "Input scores array is null");
    SCL_C_API_CHECK_NULL(labels, "Output labels array is null");
    SCL_C_API_CHECK_NULL(confidence, "Output confidence array is null");
    
    SCL_C_API_TRY
        scl::kernel::annotation::assign_from_marker_scores(
            reinterpret_cast<const Real*>(scores),
            n_cells,
            n_types,
            Array<Index>(
                reinterpret_cast<Index*>(labels),
                static_cast<Size>(n_cells)
            ),
            Array<Real>(
                reinterpret_cast<Real*>(confidence),
                static_cast<Size>(n_cells)
            )
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Consensus Annotation
// =============================================================================

SCL_EXPORT scl_error_t scl_annotation_consensus_annotation(
    const scl_index_t* const* predictions,
    const scl_real_t* const* confidences,
    const scl_index_t n_methods,
    const scl_index_t n_cells,
    const scl_index_t n_types,
    scl_index_t* consensus_labels,
    scl_real_t* consensus_confidence) {
    
    SCL_C_API_CHECK_NULL(predictions, "Predictions array is null");
    SCL_C_API_CHECK_NULL(consensus_labels, "Output consensus labels array is null");
    SCL_C_API_CHECK_NULL(consensus_confidence, "Output consensus confidence array is null");
    
    SCL_C_API_TRY
        // PERFORMANCE: C array cast - safe due to compatible layout
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        const auto* const* pred_arrays = 
            reinterpret_cast<const Index* const*>(predictions);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        const Real* const* conf_arrays = confidences ?
            reinterpret_cast<const Real* const*>(confidences) : nullptr;
        
        scl::kernel::annotation::consensus_annotation(
            pred_arrays,
            conf_arrays,
            n_methods,
            n_cells,
            n_types,
            Array<Index>(
                reinterpret_cast<Index*>(consensus_labels),
                static_cast<Size>(n_cells)
            ),
            Array<Real>(
                reinterpret_cast<Real*>(consensus_confidence),
                static_cast<Size>(n_cells)
            )
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Novel Type Detection
// =============================================================================

SCL_EXPORT scl_error_t scl_annotation_detect_novel_types(
    scl_sparse_t query_expression,
    const scl_real_t* confidence_scores,
    const scl_index_t n_query,
    const scl_real_t threshold,
    scl_bool_t* is_novel) {
    
    SCL_C_API_CHECK_NULL(query_expression, "Query expression matrix is null");
    SCL_C_API_CHECK_NULL(confidence_scores, "Confidence scores array is null");
    SCL_C_API_CHECK_NULL(is_novel, "Output is_novel array is null");
    
    SCL_C_API_TRY
        query_expression->visit([&](auto& query) {
            scl::kernel::annotation::detect_novel_types(
                query,
                Array<const Real>(
                    reinterpret_cast<const Real*>(confidence_scores),
                    static_cast<Size>(n_query)
                ),
                n_query,
                static_cast<Real>(threshold),
                Array<bool>(
                    reinterpret_cast<bool*>(is_novel),
                    static_cast<Size>(n_query)
                )
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_annotation_detect_novel_by_distance(
    scl_sparse_t query_expression,
    const scl_real_t* reference_profiles,
    const scl_index_t* assigned_labels,
    const scl_index_t n_query,
    const scl_index_t n_types,
    const scl_index_t n_genes,
    const scl_real_t distance_threshold,
    scl_bool_t* is_novel,
    scl_real_t* distance_to_assigned) {
    
    SCL_C_API_CHECK_NULL(query_expression, "Query expression matrix is null");
    SCL_C_API_CHECK_NULL(reference_profiles, "Reference profiles array is null");
    SCL_C_API_CHECK_NULL(assigned_labels, "Assigned labels array is null");
    SCL_C_API_CHECK_NULL(is_novel, "Output is_novel array is null");
    
    SCL_C_API_TRY
        query_expression->visit([&](auto& query) {
            scl::kernel::annotation::detect_novel_types_by_distance(
                query,
                reinterpret_cast<const Real*>(reference_profiles),
                Array<const Index>(
                    reinterpret_cast<const Index*>(assigned_labels),
                    static_cast<Size>(n_query)
                ),
                n_query,
                n_types,
                n_genes,
                static_cast<Real>(distance_threshold),
                Array<bool>(
                    reinterpret_cast<bool*>(is_novel),
                    static_cast<Size>(n_query)
                ),
                distance_to_assigned ? Array<Real>(
                    reinterpret_cast<Real*>(distance_to_assigned),
                    static_cast<Size>(n_query)
                ) : Array<Real>(nullptr, 0)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Label Propagation
// =============================================================================

SCL_EXPORT scl_error_t scl_annotation_label_propagation(
    scl_sparse_t neighbor_graph,
    const scl_index_t* initial_labels,
    const scl_index_t n_cells,
    const scl_index_t n_types,
    const scl_index_t max_iter,
    scl_index_t* final_labels,
    scl_real_t* label_confidence) {
    
    SCL_C_API_CHECK_NULL(neighbor_graph, "Neighbor graph is null");
    SCL_C_API_CHECK_NULL(initial_labels, "Initial labels array is null");
    SCL_C_API_CHECK_NULL(final_labels, "Output final labels array is null");
    SCL_C_API_CHECK_NULL(label_confidence, "Output label confidence array is null");
    
    SCL_C_API_TRY
        neighbor_graph->visit([&](auto& graph) {
            scl::kernel::annotation::label_propagation(
                graph,
                Array<const Index>(
                    reinterpret_cast<const Index*>(initial_labels),
                    static_cast<Size>(n_cells)
                ),
                n_cells,
                n_types,
                max_iter,
                Array<Index>(
                    reinterpret_cast<Index*>(final_labels),
                    static_cast<Size>(n_cells)
                ),
                Array<Real>(
                    reinterpret_cast<Real*>(label_confidence),
                    static_cast<Size>(n_cells)
                )
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Quality Metrics
// =============================================================================

SCL_EXPORT scl_error_t scl_annotation_quality_metrics(
    const scl_index_t* predicted_labels,
    const scl_index_t* true_labels,
    const scl_index_t n_cells,
    const scl_index_t n_types,
    scl_real_t* accuracy,
    scl_real_t* macro_f1,
    scl_real_t* per_class_f1) {
    
    SCL_C_API_CHECK_NULL(predicted_labels, "Predicted labels array is null");
    SCL_C_API_CHECK_NULL(true_labels, "True labels array is null");
    SCL_C_API_CHECK_NULL(accuracy, "Output accuracy pointer is null");
    SCL_C_API_CHECK_NULL(macro_f1, "Output macro_f1 pointer is null");
    
    SCL_C_API_TRY
        Real acc = Real(0);
        Real f1 = Real(0);
        
        scl::kernel::annotation::annotation_quality_metrics(
            Array<const Index>(
                reinterpret_cast<const Index*>(predicted_labels),
                static_cast<Size>(n_cells)
            ),
            Array<const Index>(
                reinterpret_cast<const Index*>(true_labels),
                static_cast<Size>(n_cells)
            ),
            n_cells,
            n_types,
            acc,
            f1,
            per_class_f1 ? reinterpret_cast<Real*>(per_class_f1) : nullptr
        );
        
        *accuracy = static_cast<scl_real_t>(acc);
        *macro_f1 = static_cast<scl_real_t>(f1);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_annotation_confusion_matrix(
    const scl_index_t* predicted_labels,
    const scl_index_t* true_labels,
    const scl_index_t n_cells,
    const scl_index_t n_types,
    scl_index_t* confusion) {
    
    SCL_C_API_CHECK_NULL(predicted_labels, "Predicted labels array is null");
    SCL_C_API_CHECK_NULL(true_labels, "True labels array is null");
    SCL_C_API_CHECK_NULL(confusion, "Output confusion matrix is null");
    
    SCL_C_API_TRY
        scl::kernel::annotation::confusion_matrix(
            Array<const Index>(
                reinterpret_cast<const Index*>(predicted_labels),
                static_cast<Size>(n_cells)
            ),
            Array<const Index>(
                reinterpret_cast<const Index*>(true_labels),
                static_cast<Size>(n_cells)
            ),
            n_cells,
            n_types,
            reinterpret_cast<Index*>(confusion)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_annotation_entropy(
    const scl_real_t* type_probabilities,
    const scl_index_t n_cells,
    const scl_index_t n_types,
    scl_real_t* entropy) {
    
    SCL_C_API_CHECK_NULL(type_probabilities, "Type probabilities array is null");
    SCL_C_API_CHECK_NULL(entropy, "Output entropy array is null");
    
    SCL_C_API_TRY
        scl::kernel::annotation::annotation_entropy(
            reinterpret_cast<const Real*>(type_probabilities),
            n_cells,
            n_types,
            Array<Real>(
                reinterpret_cast<Real*>(entropy),
                static_cast<Size>(n_cells)
            )
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
