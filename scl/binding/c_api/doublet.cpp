// =============================================================================
// FILE: scl/binding/c_api/doublet/doublet.cpp
// BRIEF: C API implementation for doublet detection
// =============================================================================

#include "scl/binding/c_api/doublet.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/doublet.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

namespace {
    using namespace scl::kernel::doublet;
}

// =============================================================================
// Doublet Simulation
// =============================================================================

scl_error_t scl_doublet_simulate_doublets(
    scl_sparse_t X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_doublets,
    scl_real_t* doublet_profiles,
    uint64_t seed
) {
    if (!X || !doublet_profiles) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        X->visit([&](auto& mat) {
            simulate_doublets(mat, n_cells, n_genes, n_doublets,
                            reinterpret_cast<scl::Real*>(doublet_profiles), seed);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// KNN Score Computation
// =============================================================================

scl_error_t scl_doublet_compute_knn_scores(
    scl_sparse_t X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    const scl_real_t* doublet_profiles,
    scl_index_t n_doublets,
    scl_index_t k_neighbors,
    scl_real_t* doublet_scores
) {
    if (!X || !doublet_profiles || !doublet_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<scl::Real> scores_arr(
            reinterpret_cast<scl::Real*>(doublet_scores),
            static_cast<scl::Size>(n_cells)
        );
        X->visit([&](auto& mat) {
            compute_knn_doublet_scores(mat, n_cells, n_genes,
                reinterpret_cast<const scl::Real*>(doublet_profiles),
                n_doublets, k_neighbors, scores_arr);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_doublet_compute_knn_scores_pca(
    const scl_real_t* cell_embeddings,
    scl_index_t n_cells,
    scl_index_t n_dims,
    const scl_real_t* doublet_embeddings,
    scl_index_t n_doublets,
    scl_index_t k_neighbors,
    scl_real_t* doublet_scores
) {
    if (!cell_embeddings || !doublet_embeddings || !doublet_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<scl::Real> scores_arr(
            reinterpret_cast<scl::Real*>(doublet_scores),
            static_cast<scl::Size>(n_cells)
        );
        compute_knn_doublet_scores_pca(
            reinterpret_cast<const scl::Real*>(cell_embeddings),
            n_cells, n_dims,
            reinterpret_cast<const scl::Real*>(doublet_embeddings),
            n_doublets, k_neighbors, scores_arr);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Scrublet
// =============================================================================

scl_error_t scl_doublet_scrublet_scores(
    scl_sparse_t X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* scores,
    scl_index_t n_simulated,
    scl_index_t k_neighbors,
    uint64_t seed
) {
    if (!X || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<scl::Real> scores_arr(
            reinterpret_cast<scl::Real*>(scores),
            static_cast<scl::Size>(n_cells)
        );
        X->visit([&](auto& mat) {
            scrublet_scores(mat, n_cells, n_genes, scores_arr,
                          n_simulated, k_neighbors, seed);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// DoubletFinder
// =============================================================================

scl_error_t scl_doublet_doubletfinder_pann(
    const scl_real_t* cell_embeddings,
    scl_index_t n_cells,
    scl_index_t n_dims,
    const scl_real_t* doublet_embeddings,
    scl_index_t n_doublets,
    scl_real_t pK,
    scl_real_t* pann_scores
) {
    if (!cell_embeddings || !doublet_embeddings || !pann_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<scl::Real> scores_arr(
            reinterpret_cast<scl::Real*>(pann_scores),
            static_cast<scl::Size>(n_cells)
        );
        doubletfinder_pann(
            reinterpret_cast<const scl::Real*>(cell_embeddings),
            n_cells, n_dims,
            reinterpret_cast<const scl::Real*>(doublet_embeddings),
            n_doublets, pK, scores_arr);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Threshold Estimation
// =============================================================================

scl_error_t scl_doublet_estimate_threshold(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_real_t expected_doublet_rate,
    scl_real_t* threshold_out
) {
    if (!scores || !threshold_out) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> scores_arr(
            reinterpret_cast<const scl::Real*>(scores),
            n_scores
        );
        *threshold_out = static_cast<scl_real_t>(
            estimate_threshold(scores_arr, expected_doublet_rate)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_doublet_detect_bimodal_threshold(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_index_t n_bins,
    scl_real_t* threshold_out
) {
    if (!scores || !threshold_out) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> scores_arr(
            reinterpret_cast<const scl::Real*>(scores),
            n_scores
        );
        *threshold_out = static_cast<scl_real_t>(
            detect_bimodal_threshold(scores_arr, n_bins)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Doublet Calling
// =============================================================================

scl_error_t scl_doublet_call_doublets(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_real_t threshold,
    int* is_doublet,
    scl_index_t* n_doublets_out
) {
    if (!scores || !is_doublet || !n_doublets_out) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> scores_arr(
            reinterpret_cast<const scl::Real*>(scores),
            n_scores
        );
        scl::Array<bool> is_doublet_arr(
            reinterpret_cast<bool*>(is_doublet),
            n_scores
        );
        *n_doublets_out = call_doublets(scores_arr, threshold, is_doublet_arr);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Full Detection Pipeline
// =============================================================================

scl_error_t scl_doublet_detect_doublets(
    scl_sparse_t X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* scores,
    int* is_doublet,
    scl_doublet_method_t method,
    scl_real_t expected_rate,
    scl_index_t k_neighbors,
    uint64_t seed,
    scl_index_t* n_doublets_out
) {
    if (!X || !scores || !is_doublet || !n_doublets_out) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<scl::Real> scores_arr(
            reinterpret_cast<scl::Real*>(scores),
            static_cast<scl::Size>(n_cells)
        );
        scl::Array<bool> is_doublet_arr(
            reinterpret_cast<bool*>(is_doublet),
            static_cast<scl::Size>(n_cells)
        );
        DoubletMethod m = (method == SCL_DOUBLET_METHOD_DOUBLETFINDER) ?
            DoubletMethod::DoubletFinder :
            (method == SCL_DOUBLET_METHOD_HYBRID) ?
            DoubletMethod::Hybrid : DoubletMethod::Scrublet;

        X->visit([&](auto& mat) {
            *n_doublets_out = detect_doublets(mat, n_cells, n_genes,
                scores_arr, is_doublet_arr, m, expected_rate,
                k_neighbors, seed);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

scl_error_t scl_doublet_get_singlet_indices(
    const int* is_doublet,
    scl_size_t n_cells,
    scl_index_t* singlet_indices,
    scl_size_t max_indices,
    scl_index_t* n_singlets_out
) {
    if (!is_doublet || !singlet_indices || !n_singlets_out) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const bool> is_doublet_arr(
            reinterpret_cast<const bool*>(is_doublet),
            n_cells
        );
        scl::Array<scl::Index> singlet_arr(
            reinterpret_cast<scl::Index*>(singlet_indices),
            max_indices
        );
        *n_singlets_out = get_singlet_indices(is_doublet_arr, singlet_arr);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_doublet_score_stats(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_real_t* mean_out,
    scl_real_t* std_dev_out,
    scl_real_t* median_out
) {
    if (!scores || !mean_out || !std_dev_out || !median_out) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> scores_arr(
            reinterpret_cast<const scl::Real*>(scores),
            n_scores
        );
        scl::Real mean, std_dev, median;
        doublet_score_stats(scores_arr, &mean, &std_dev, &median);
        *mean_out = static_cast<scl_real_t>(mean);
        *std_dev_out = static_cast<scl_real_t>(std_dev);
        *median_out = static_cast<scl_real_t>(median);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
