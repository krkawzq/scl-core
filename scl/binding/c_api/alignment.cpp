// =============================================================================
// FILE: scl/binding/c_api/alignment.cpp
// BRIEF: C API implementation for multi-modal data alignment
// =============================================================================

#include "alignment.h"
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/kernel/alignment.hpp"

namespace {
    inline scl_error_t catch_exception() {
        try {
            throw;
        } catch (const scl::DimensionError&) {
            return SCL_ERROR_DIMENSION_MISMATCH;
        } catch (const scl::ValueError&) {
            return SCL_ERROR_INVALID_ARGUMENT;
        } catch (const scl::Exception& e) {
            return static_cast<scl_error_t>(e.code());
        } catch (...) {
            return SCL_ERROR_UNKNOWN;
        }
    }

    template <typename T>
    inline scl::Sparse<T, true>* get_matrix(scl_sparse_matrix_t handle) {
        return reinterpret_cast<scl::Sparse<T, true>*>(handle);
    }
}

extern "C" scl_error_t scl_mnn_pairs_f32_csr(
    scl_sparse_matrix_t data1,
    scl_sparse_matrix_t data2,
    scl_index_t k,
    scl_index_t* mnn_cell1,
    scl_index_t* mnn_cell2,
    scl_size_t max_pairs,
    scl_size_t* n_pairs
) {
    try {
        auto* m1 = get_matrix<float>(data1);
        auto* m2 = get_matrix<float>(data2);
        if (!m1 || !m2 || !mnn_cell1 || !mnn_cell2 || !n_pairs) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Size n_pairs_val = 0;
        scl::kernel::alignment::mnn_pairs(*m1, *m2, k, mnn_cell1, mnn_cell2, n_pairs_val);
        *n_pairs = n_pairs_val;
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_mnn_pairs_f64_csr(
    scl_sparse_matrix_t data1,
    scl_sparse_matrix_t data2,
    scl_index_t k,
    scl_index_t* mnn_cell1,
    scl_index_t* mnn_cell2,
    scl_size_t max_pairs,
    scl_size_t* n_pairs
) {
    try {
        auto* m1 = get_matrix<double>(data1);
        auto* m2 = get_matrix<double>(data2);
        if (!m1 || !m2 || !mnn_cell1 || !mnn_cell2 || !n_pairs) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Size n_pairs_val = 0;
        scl::kernel::alignment::mnn_pairs(*m1, *m2, k, mnn_cell1, mnn_cell2, n_pairs_val);
        *n_pairs = n_pairs_val;
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_find_anchors_f32_csr(
    scl_sparse_matrix_t data1,
    scl_sparse_matrix_t data2,
    scl_index_t k,
    scl_index_t* anchor_cell1,
    scl_index_t* anchor_cell2,
    scl_real_t* anchor_scores,
    scl_size_t max_anchors,
    scl_size_t* n_anchors
) {
    try {
        auto* m1 = get_matrix<float>(data1);
        auto* m2 = get_matrix<float>(data2);
        if (!m1 || !m2 || !anchor_cell1 || !anchor_cell2 || !anchor_scores || !n_anchors) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Size n_anchors_val = 0;
        scl::kernel::alignment::find_anchors(*m1, *m2, k, anchor_cell1, anchor_cell2, anchor_scores, n_anchors_val);
        *n_anchors = n_anchors_val;
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_find_anchors_f64_csr(
    scl_sparse_matrix_t data1,
    scl_sparse_matrix_t data2,
    scl_index_t k,
    scl_index_t* anchor_cell1,
    scl_index_t* anchor_cell2,
    double* anchor_scores,
    scl_size_t max_anchors,
    scl_size_t* n_anchors
) {
    try {
        auto* m1 = get_matrix<double>(data1);
        auto* m2 = get_matrix<double>(data2);
        if (!m1 || !m2 || !anchor_cell1 || !anchor_cell2 || !anchor_scores || !n_anchors) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Size n_anchors_val = 0;
        scl::kernel::alignment::find_anchors(*m1, *m2, k, anchor_cell1, anchor_cell2, anchor_scores, n_anchors_val);
        *n_anchors = n_anchors_val;
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_transfer_labels(
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
    try {
        if (!anchor_cell1 || !anchor_cell2 || !anchor_weights || !source_labels || !target_labels || !transfer_confidence) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> source_arr(source_labels, n_source);
        scl::Array<scl::Index> target_arr(target_labels, n_target);
        scl::Array<scl::Real> conf_arr(transfer_confidence, n_target);

        scl::kernel::alignment::transfer_labels(
            anchor_cell1, anchor_cell2, anchor_weights, n_anchors,
            source_arr, n_target, target_arr, conf_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_integration_score_f32_csr(
    scl_sparse_matrix_t integrated_data,
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_sparse_matrix_t neighbors,
    scl_real_t* score
) {
    try {
        auto* data = get_matrix<float>(integrated_data);
        auto* neigh = get_matrix<scl::Index>(neighbors);
        if (!data || !neigh || !batch_labels || !score) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> batch_arr(batch_labels, n_cells);
        *score = scl::kernel::alignment::integration_score(*data, batch_arr, *neigh);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_integration_score_f64_csr(
    scl_sparse_matrix_t integrated_data,
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_sparse_matrix_t neighbors,
    double* score
) {
    try {
        auto* data = get_matrix<double>(integrated_data);
        auto* neigh = get_matrix<scl::Index>(neighbors);
        if (!data || !neigh || !batch_labels || !score) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> batch_arr(batch_labels, n_cells);
        *score = scl::kernel::alignment::integration_score(*data, batch_arr, *neigh);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_batch_mixing(
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_sparse_matrix_t neighbors,
    scl_real_t* mixing_scores
) {
    try {
        auto* neigh = get_matrix<scl::Index>(neighbors);
        if (!neigh || !batch_labels || !mixing_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> batch_arr(batch_labels, n_cells);
        scl::Array<scl::Real> scores_arr(mixing_scores, n_cells);
        scl::kernel::alignment::batch_mixing(batch_arr, *neigh, scores_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_kbet_score(
    scl_sparse_matrix_t neighbors,
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_real_t* score
) {
    try {
        auto* neigh = get_matrix<scl::Index>(neighbors);
        if (!neigh || !batch_labels || !score) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> batch_arr(batch_labels, n_cells);
        *score = scl::kernel::alignment::kbet_score(*neigh, batch_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}
