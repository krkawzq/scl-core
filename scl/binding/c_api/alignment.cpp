// =============================================================================
// FILE: scl/binding/c_api/alignment.cpp
// BRIEF: C API implementation for multi-modal data alignment
// =============================================================================

#include "scl/binding/c_api/alignment.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/alignment.hpp"
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

scl_error_t scl_alignment_mnn_pairs(
    scl_sparse_t data1,
    scl_sparse_t data2,
    scl_index_t k,
    scl_index_t* mnn_cell1,
    scl_index_t* mnn_cell2,
    scl_size_t* n_pairs
) {
    if (!data1 || !data2 || !mnn_cell1 || !mnn_cell2 || !n_pairs) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper1;
        scl::binding::SparseWrapper* wrapper2;
        scl_error_t err1 = get_sparse_matrix(data1, wrapper1);
        scl_error_t err2 = get_sparse_matrix(data2, wrapper2);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        scl::Size n_pairs_result = 0;
        wrapper1->visit([&](auto& m1) {
            wrapper2->visit([&](auto& m2) {
                scl::kernel::alignment::mnn_pairs(
                    m1, m2,
                    static_cast<scl::Index>(k),
                    reinterpret_cast<scl::Index*>(mnn_cell1),
                    reinterpret_cast<scl::Index*>(mnn_cell2),
                    n_pairs_result
                );
            });
        });
        *n_pairs = static_cast<scl_size_t>(n_pairs_result);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_alignment_find_anchors(
    scl_sparse_t data1,
    scl_sparse_t data2,
    scl_index_t k,
    scl_index_t* anchor_cell1,
    scl_index_t* anchor_cell2,
    scl_real_t* anchor_scores,
    scl_size_t* n_anchors
) {
    if (!data1 || !data2 || !anchor_cell1 || !anchor_cell2 || !anchor_scores || !n_anchors) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper1;
        scl::binding::SparseWrapper* wrapper2;
        scl_error_t err1 = get_sparse_matrix(data1, wrapper1);
        scl_error_t err2 = get_sparse_matrix(data2, wrapper2);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        scl::Size n_anchors_result = 0;
        wrapper1->visit([&](auto& m1) {
            wrapper2->visit([&](auto& m2) {
                scl::kernel::alignment::find_anchors(
                    m1, m2,
                    static_cast<scl::Index>(k),
                    reinterpret_cast<scl::Index*>(anchor_cell1),
                    reinterpret_cast<scl::Index*>(anchor_cell2),
                    reinterpret_cast<scl::Real*>(anchor_scores),
                    n_anchors_result
                );
            });
        });
        *n_anchors = static_cast<scl_size_t>(n_anchors_result);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_alignment_transfer_labels(
    const scl_index_t* anchor_cell1,
    const scl_index_t* anchor_cell2,
    const scl_real_t* anchor_weights,
    scl_size_t n_anchors,
    const scl_index_t* source_labels,
    scl_size_t n_source,
    scl_size_t n_target,
    scl_index_t* target_labels,
    scl_real_t* transfer_confidence
) {
    if (!anchor_cell1 || !anchor_cell2 || !anchor_weights || !source_labels ||
        !target_labels || !transfer_confidence) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::kernel::alignment::transfer_labels(
            reinterpret_cast<const scl::Index*>(anchor_cell1),
            reinterpret_cast<const scl::Index*>(anchor_cell2),
            reinterpret_cast<const scl::Real*>(anchor_weights),
            static_cast<scl::Size>(n_anchors),
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(source_labels),
                static_cast<scl::Size>(n_source)
            ),
            static_cast<scl::Size>(n_target),
            scl::Array<scl::Index>(
                reinterpret_cast<scl::Index*>(target_labels),
                static_cast<scl::Size>(n_target)
            ),
            scl::Array<scl::Real>(
                reinterpret_cast<scl::Real*>(transfer_confidence),
                static_cast<scl::Size>(n_target)
            )
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_alignment_integration_score(
    scl_sparse_t integrated_data,
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_sparse_t neighbors,
    scl_real_t* score
) {
    if (!integrated_data || !batch_labels || !neighbors || !score) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper_data;
        scl::binding::SparseWrapper* wrapper_neighbors;
        scl_error_t err1 = get_sparse_matrix(integrated_data, wrapper_data);
        scl_error_t err2 = get_sparse_matrix(neighbors, wrapper_neighbors);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        scl::Real score_result = scl::Real(0);
        wrapper_data->visit([&](auto& data) {
            using DataType = std::remove_reference_t<decltype(data)>;
            using T = typename DataType::ValueType;
            constexpr bool IsCSR_Data = DataType::is_csr;
            
            wrapper_neighbors->visit([&](auto& neigh) {
                using NeighborType = std::remove_reference_t<decltype(neigh)>;
                constexpr bool IsCSR_Neighbors = NeighborType::is_csr;
                using NeighborValueType = typename NeighborType::ValueType;
                
                // Cast neighbor matrix to Index type if needed
                if constexpr (std::is_same_v<NeighborValueType, scl::Index>) {
                    score_result = scl::kernel::alignment::integration_score<T, IsCSR_Data, IsCSR_Neighbors>(
                        data,
                        scl::Array<const scl::Index>(
                            reinterpret_cast<const scl::Index*>(batch_labels),
                            static_cast<scl::Size>(n_cells)
                        ),
                        neigh
                    );
                } else {
                    // Neighbor matrix has wrong value type, return error
                    score_result = scl::Real(0);
                }
            });
        });
        *score = static_cast<scl_real_t>(score_result);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_alignment_batch_mixing(
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_sparse_t neighbors,
    scl_real_t* mixing_scores
) {
    if (!batch_labels || !neighbors || !mixing_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(neighbors, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& neigh) {
            using NeighborType = std::remove_reference_t<decltype(neigh)>;
            constexpr bool IsCSR = NeighborType::is_csr;
            using NeighborValueType = typename NeighborType::ValueType;
            
            if constexpr (std::is_same_v<NeighborValueType, scl::Index>) {
                scl::kernel::alignment::batch_mixing<IsCSR>(
                    scl::Array<const scl::Index>(
                        reinterpret_cast<const scl::Index*>(batch_labels),
                        static_cast<scl::Size>(n_cells)
                    ),
                    neigh,
                    scl::Array<scl::Real>(
                        reinterpret_cast<scl::Real*>(mixing_scores),
                        static_cast<scl::Size>(n_cells)
                    )
                );
            }
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_alignment_compute_correction_vectors(
    scl_sparse_t data1,
    scl_sparse_t data2,
    const scl_index_t* mnn_cell1,
    const scl_index_t* mnn_cell2,
    scl_size_t n_pairs,
    scl_real_t* correction_vectors,
    scl_size_t n_features
) {
    if (!data1 || !data2 || !mnn_cell1 || !mnn_cell2 || !correction_vectors) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper1;
        scl::binding::SparseWrapper* wrapper2;
        scl_error_t err1 = get_sparse_matrix(data1, wrapper1);
        scl_error_t err2 = get_sparse_matrix(data2, wrapper2);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        wrapper1->visit([&](auto& m1) {
            wrapper2->visit([&](auto& m2) {
                scl::kernel::alignment::compute_correction_vectors(
                    m1, m2,
                    reinterpret_cast<const scl::Index*>(mnn_cell1),
                    reinterpret_cast<const scl::Index*>(mnn_cell2),
                    static_cast<scl::Size>(n_pairs),
                    reinterpret_cast<scl::Real*>(correction_vectors),
                    static_cast<scl::Size>(n_features)
                );
            });
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_alignment_smooth_correction_vectors(
    scl_sparse_t data2,
    scl_real_t* correction_vectors,
    scl_size_t n_features,
    scl_real_t sigma
) {
    if (!data2 || !correction_vectors) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(data2, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& m) {
            scl::kernel::alignment::smooth_correction_vectors(
                m,
                reinterpret_cast<scl::Real*>(correction_vectors),
                static_cast<scl::Size>(n_features),
                static_cast<scl::Real>(sigma)
            );
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_alignment_cca_projection(
    scl_sparse_t data1,
    scl_sparse_t data2,
    scl_size_t n_components,
    scl_real_t* projection1,
    scl_real_t* projection2
) {
    if (!data1 || !data2 || !projection1 || !projection2) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper1;
        scl::binding::SparseWrapper* wrapper2;
        scl_error_t err1 = get_sparse_matrix(data1, wrapper1);
        scl_error_t err2 = get_sparse_matrix(data2, wrapper2);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        wrapper1->visit([&](auto& m1) {
            wrapper2->visit([&](auto& m2) {
                scl::kernel::alignment::cca_projection(
                    m1, m2,
                    static_cast<scl::Size>(n_components),
                    reinterpret_cast<scl::Real*>(projection1),
                    reinterpret_cast<scl::Real*>(projection2)
                );
            });
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_alignment_kbet_score(
    scl_sparse_t neighbors,
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_real_t* score
) {
    if (!neighbors || !batch_labels || !score) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(neighbors, wrapper);
        if (err != SCL_OK) return err;

        scl::Real score_result = scl::Real(0);
        wrapper->visit([&](auto& neigh) {
            using NeighborType = std::remove_reference_t<decltype(neigh)>;
            constexpr bool IsCSR = NeighborType::is_csr;
            using NeighborValueType = typename NeighborType::ValueType;
            
            if constexpr (std::is_same_v<NeighborValueType, scl::Index>) {
                score_result = scl::kernel::alignment::kbet_score<IsCSR>(
                    neigh,
                    scl::Array<const scl::Index>(
                        reinterpret_cast<const scl::Index*>(batch_labels),
                        static_cast<scl::Size>(n_cells)
                    )
                );
            }
        });
        *score = static_cast<scl_real_t>(score_result);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
