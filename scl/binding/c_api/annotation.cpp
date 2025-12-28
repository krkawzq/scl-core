// =============================================================================
// FILE: scl/binding/c_api/annotation.cpp
// BRIEF: C API implementation for cell type annotation
// =============================================================================

#include "scl/binding/c_api/annotation.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/annotation.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

static scl_error_t get_sparse_matrix(
    scl_sparse_t handle,
    scl::binding::SparseWrapper*& wrapper
) {
    if (!handle) {
        return SCL_ERROR_NULL_POINTER;
    }
    wrapper = static_cast<scl::binding::SparseWrapper*>(handle);
    if (!wrapper->valid()) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    return SCL_OK;
}

scl_error_t scl_annotation_reference_mapping(
    scl_sparse_t query_expression,
    scl_sparse_t reference_expression,
    const scl_index_t* reference_labels,
    scl_size_t n_ref,
    scl_sparse_t query_to_ref_neighbors,
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t* query_labels,
    scl_real_t* confidence_scores
) {
    if (!query_expression || !reference_expression || !reference_labels ||
        !query_to_ref_neighbors || !query_labels || !confidence_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper_query;
        scl::binding::SparseWrapper* wrapper_ref;
        scl::binding::SparseWrapper* wrapper_neighbors;
        scl_error_t err1 = get_sparse_matrix(query_expression, wrapper_query);
        scl_error_t err2 = get_sparse_matrix(reference_expression, wrapper_ref);
        scl_error_t err3 = get_sparse_matrix(query_to_ref_neighbors, wrapper_neighbors);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;
        if (err3 != SCL_OK) return err3;

        wrapper_query->visit([&](auto& query) {
            wrapper_ref->visit([&](auto& ref) {
                wrapper_neighbors->visit([&](auto& neighbors) {
                    scl::kernel::annotation::reference_mapping(
                        query, ref,
                        scl::Array<const scl::Index>(
                            reinterpret_cast<const scl::Index*>(reference_labels),
                            static_cast<scl::Size>(n_ref)
                        ),
                        neighbors,
                        static_cast<scl::Index>(n_query),
                        static_cast<scl::Index>(n_ref),
                        static_cast<scl::Index>(n_types),
                        scl::Array<scl::Index>(
                            reinterpret_cast<scl::Index*>(query_labels),
                            static_cast<scl::Size>(n_query)
                        ),
                        scl::Array<scl::Real>(
                            reinterpret_cast<scl::Real*>(confidence_scores),
                            static_cast<scl::Size>(n_query)
                        )
                    );
                });
            });
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_annotation_correlation_assignment(
    scl_sparse_t query_expression,
    scl_sparse_t reference_profiles,
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t n_genes,
    scl_index_t* assigned_labels,
    scl_real_t* correlation_scores,
    scl_real_t* all_correlations
) {
    if (!query_expression || !reference_profiles || !assigned_labels || !correlation_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper_query;
        scl::binding::SparseWrapper* wrapper_profiles;
        scl_error_t err1 = get_sparse_matrix(query_expression, wrapper_query);
        scl_error_t err2 = get_sparse_matrix(reference_profiles, wrapper_profiles);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        wrapper_query->visit([&](auto& query) {
            wrapper_profiles->visit([&](auto& profiles) {
                scl::kernel::annotation::correlation_assignment(
                    query, profiles,
                    static_cast<scl::Index>(n_query),
                    static_cast<scl::Index>(n_types),
                    static_cast<scl::Index>(n_genes),
                    scl::Array<scl::Index>(
                        reinterpret_cast<scl::Index*>(assigned_labels),
                        static_cast<scl::Size>(n_query)
                    ),
                    scl::Array<scl::Real>(
                        reinterpret_cast<scl::Real*>(correlation_scores),
                        static_cast<scl::Size>(n_query)
                    ),
                    all_correlations ? scl::Array<scl::Real>(
                        reinterpret_cast<scl::Real*>(all_correlations),
                        static_cast<scl::Size>(n_query) * static_cast<scl::Size>(n_types)
                    ) : scl::Array<scl::Real>(nullptr, 0)
                );
            });
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_annotation_build_reference_profiles(
    scl_sparse_t expression,
    const scl_index_t* labels,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    scl_real_t* profiles
) {
    if (!expression || !labels || !profiles) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(expression, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& expr) {
            scl::kernel::annotation::build_reference_profiles(
                expr,
                scl::Array<const scl::Index>(
                    reinterpret_cast<const scl::Index*>(labels),
                    static_cast<scl::Size>(n_cells)
                ),
                static_cast<scl::Index>(n_cells),
                static_cast<scl::Index>(n_genes),
                static_cast<scl::Index>(n_types),
                reinterpret_cast<scl::Real*>(profiles)
            );
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_annotation_marker_gene_score(
    scl_sparse_t expression,
    const scl_index_t* const* marker_genes,
    const scl_index_t* marker_counts,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    scl_real_t* scores,
    int normalize
) {
    if (!expression || !marker_genes || !marker_counts || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(expression, wrapper);
        if (err != SCL_OK) return err;

        // Convert C array of arrays to C++ format
        const scl::Index* const* marker_arrays = 
            reinterpret_cast<const scl::Index* const*>(marker_genes);

        wrapper->visit([&](auto& expr) {
            scl::kernel::annotation::marker_gene_score(
                expr,
                marker_arrays,
                reinterpret_cast<const scl::Index*>(marker_counts),
                static_cast<scl::Index>(n_cells),
                static_cast<scl::Index>(n_genes),
                static_cast<scl::Index>(n_types),
                reinterpret_cast<scl::Real*>(scores),
                normalize != 0
            );
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_annotation_assign_from_marker_scores(
    const scl_real_t* scores,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_index_t* labels,
    scl_real_t* confidence
) {
    if (!scores || !labels || !confidence) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::kernel::annotation::assign_from_marker_scores(
            reinterpret_cast<const scl::Real*>(scores),
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_types),
            scl::Array<scl::Index>(
                reinterpret_cast<scl::Index*>(labels),
                static_cast<scl::Size>(n_cells)
            ),
            scl::Array<scl::Real>(
                reinterpret_cast<scl::Real*>(confidence),
                static_cast<scl::Size>(n_cells)
            )
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_annotation_consensus_annotation(
    const scl_index_t* const* predictions,
    const scl_real_t* const* confidences,
    scl_index_t n_methods,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_index_t* consensus_labels,
    scl_real_t* consensus_confidence
) {
    if (!predictions || !consensus_labels || !consensus_confidence) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        const scl::Index* const* pred_arrays = 
            reinterpret_cast<const scl::Index* const*>(predictions);
        const scl::Real* const* conf_arrays = confidences ?
            reinterpret_cast<const scl::Real* const*>(confidences) : nullptr;

        scl::kernel::annotation::consensus_annotation(
            pred_arrays,
            conf_arrays,
            static_cast<scl::Index>(n_methods),
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_types),
            scl::Array<scl::Index>(
                reinterpret_cast<scl::Index*>(consensus_labels),
                static_cast<scl::Size>(n_cells)
            ),
            scl::Array<scl::Real>(
                reinterpret_cast<scl::Real*>(consensus_confidence),
                static_cast<scl::Size>(n_cells)
            )
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_annotation_detect_novel_types(
    scl_sparse_t query_expression,
    const scl_real_t* confidence_scores,
    scl_index_t n_query,
    scl_real_t threshold,
    int* is_novel
) {
    if (!query_expression || !confidence_scores || !is_novel) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(query_expression, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& query) {
            scl::kernel::annotation::detect_novel_types(
                query,
                scl::Array<const scl::Real>(
                    reinterpret_cast<const scl::Real*>(confidence_scores),
                    static_cast<scl::Size>(n_query)
                ),
                static_cast<scl::Index>(n_query),
                static_cast<scl::Real>(threshold),
                scl::Array<bool>(
                    reinterpret_cast<bool*>(is_novel),
                    static_cast<scl::Size>(n_query)
                )
            );
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_annotation_label_propagation(
    scl_sparse_t neighbor_graph,
    const scl_index_t* initial_labels,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_index_t max_iter,
    scl_index_t* final_labels,
    scl_real_t* label_confidence
) {
    if (!neighbor_graph || !initial_labels || !final_labels || !label_confidence) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(neighbor_graph, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& graph) {
            scl::kernel::annotation::label_propagation(
                graph,
                scl::Array<const scl::Index>(
                    reinterpret_cast<const scl::Index*>(initial_labels),
                    static_cast<scl::Size>(n_cells)
                ),
                static_cast<scl::Index>(n_cells),
                static_cast<scl::Index>(n_types),
                static_cast<scl::Index>(max_iter),
                scl::Array<scl::Index>(
                    reinterpret_cast<scl::Index*>(final_labels),
                    static_cast<scl::Size>(n_cells)
                ),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(label_confidence),
                    static_cast<scl::Size>(n_cells)
                )
            );
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_annotation_quality_metrics(
    const scl_index_t* predicted_labels,
    const scl_index_t* true_labels,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* accuracy,
    scl_real_t* macro_f1,
    scl_real_t* per_class_f1
) {
    if (!predicted_labels || !true_labels || !accuracy || !macro_f1) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Real acc = scl::Real(0);
        scl::Real f1 = scl::Real(0);

        scl::kernel::annotation::annotation_quality_metrics(
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(predicted_labels),
                static_cast<scl::Size>(n_cells)
            ),
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(true_labels),
                static_cast<scl::Size>(n_cells)
            ),
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_types),
            acc,
            f1,
            per_class_f1 ? reinterpret_cast<scl::Real*>(per_class_f1) : nullptr
        );

        *accuracy = static_cast<scl_real_t>(acc);
        *macro_f1 = static_cast<scl_real_t>(f1);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
