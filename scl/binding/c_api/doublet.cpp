// =============================================================================
// FILE: scl/binding/c_api/doublet/doublet.cpp
// BRIEF: C API implementation for doublet detection
// =============================================================================

#include "scl/binding/c_api/doublet.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/doublet.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

[[nodiscard]] constexpr auto convert_doublet_method(
    scl_doublet_method_t method) noexcept -> scl::kernel::doublet::DoubletMethod {
    switch (method) {
        case SCL_DOUBLET_METHOD_SCRUBLET:
            return scl::kernel::doublet::DoubletMethod::Scrublet;
        case SCL_DOUBLET_METHOD_DOUBLETFINDER:
            return scl::kernel::doublet::DoubletMethod::DoubletFinder;
        case SCL_DOUBLET_METHOD_HYBRID:
            return scl::kernel::doublet::DoubletMethod::Hybrid;
        default:
            return scl::kernel::doublet::DoubletMethod::Scrublet;
    }
}

} // anonymous namespace

extern "C" {

// =============================================================================
// Doublet Simulation
// =============================================================================

SCL_EXPORT scl_error_t scl_doublet_simulate_doublets(
    scl_sparse_t X,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_index_t n_doublets,
    scl_real_t* doublet_profiles,
    const uint64_t seed) {
    
    SCL_C_API_CHECK_NULL(X, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(doublet_profiles, "Output doublet profiles array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0 && n_doublets > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        X->visit([&](auto& mat) {
            scl::kernel::doublet::simulate_doublets(
                mat, n_cells, n_genes, n_doublets,
                reinterpret_cast<Real*>(doublet_profiles), seed
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// KNN Score Computation
// =============================================================================

SCL_EXPORT scl_error_t scl_doublet_compute_knn_scores(
    scl_sparse_t X,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_real_t* doublet_profiles,
    const scl_index_t n_doublets,
    const scl_index_t k_neighbors,
    scl_real_t* doublet_scores) {
    
    SCL_C_API_CHECK_NULL(X, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(doublet_profiles, "Doublet profiles array is null");
    SCL_C_API_CHECK_NULL(doublet_scores, "Output doublet scores array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0 && n_doublets > 0 && k_neighbors > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<Real> scores_arr(
            reinterpret_cast<Real*>(doublet_scores),
            static_cast<Size>(n_cells)
        );
        
        X->visit([&](auto& mat) {
            scl::kernel::doublet::compute_knn_doublet_scores(
                mat, n_cells, n_genes,
                reinterpret_cast<const Real*>(doublet_profiles),
                n_doublets, k_neighbors, scores_arr
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_doublet_compute_knn_scores_pca(
    const scl_real_t* cell_embeddings,
    const scl_index_t n_cells,
    const scl_index_t n_dims,
    const scl_real_t* doublet_embeddings,
    const scl_index_t n_doublets,
    const scl_index_t k_neighbors,
    scl_real_t* doublet_scores) {
    
    SCL_C_API_CHECK_NULL(cell_embeddings, "Cell embeddings array is null");
    SCL_C_API_CHECK_NULL(doublet_embeddings, "Doublet embeddings array is null");
    SCL_C_API_CHECK_NULL(doublet_scores, "Output doublet scores array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_dims > 0 && n_doublets > 0 && k_neighbors > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<Real> scores_arr(
            reinterpret_cast<Real*>(doublet_scores),
            static_cast<Size>(n_cells)
        );
        
        scl::kernel::doublet::compute_knn_doublet_scores_pca(
            reinterpret_cast<const Real*>(cell_embeddings),
            n_cells, n_dims,
            reinterpret_cast<const Real*>(doublet_embeddings),
            n_doublets, k_neighbors, scores_arr
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Scrublet
// =============================================================================

SCL_EXPORT scl_error_t scl_doublet_scrublet_scores(
    scl_sparse_t X,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    scl_real_t* scores,
    const scl_index_t n_simulated,
    const scl_index_t k_neighbors,
    const uint64_t seed) {
    
    SCL_C_API_CHECK_NULL(X, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(scores, "Output scores array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0 && k_neighbors > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<Real> scores_arr(
            reinterpret_cast<Real*>(scores),
            static_cast<Size>(n_cells)
        );
        
        X->visit([&](auto& mat) {
            scl::kernel::doublet::scrublet_scores(
                mat, n_cells, n_genes, scores_arr,
                n_simulated, k_neighbors, seed
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// DoubletFinder
// =============================================================================

SCL_EXPORT scl_error_t scl_doublet_doubletfinder_pann(
    const scl_real_t* cell_embeddings,
    const scl_index_t n_cells,
    const scl_index_t n_dims,
    const scl_real_t* doublet_embeddings,
    const scl_index_t n_doublets,
    const scl_real_t pK,
    scl_real_t* pann_scores) {
    
    SCL_C_API_CHECK_NULL(cell_embeddings, "Cell embeddings array is null");
    SCL_C_API_CHECK_NULL(doublet_embeddings, "Doublet embeddings array is null");
    SCL_C_API_CHECK_NULL(pann_scores, "Output pANN scores array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_dims > 0 && n_doublets > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    SCL_C_API_CHECK(pK > 0 && pK <= 1, SCL_ERROR_INVALID_ARGUMENT,
                   "pK must be in (0, 1]");
    
    SCL_C_API_TRY
        Array<Real> scores_arr(
            reinterpret_cast<Real*>(pann_scores),
            static_cast<Size>(n_cells)
        );
        
        scl::kernel::doublet::doubletfinder_pann(
            reinterpret_cast<const Real*>(cell_embeddings),
            n_cells, n_dims,
            reinterpret_cast<const Real*>(doublet_embeddings),
            n_doublets, static_cast<Real>(pK), scores_arr
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Threshold Estimation
// =============================================================================

SCL_EXPORT scl_error_t scl_doublet_estimate_threshold(
    const scl_real_t* scores,
    const scl_size_t n_scores,
    const scl_real_t expected_doublet_rate,
    scl_real_t* threshold_out) {
    
    SCL_C_API_CHECK_NULL(scores, "Scores array is null");
    SCL_C_API_CHECK_NULL(threshold_out, "Output threshold pointer is null");
    SCL_C_API_CHECK(n_scores > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of scores must be positive");
    SCL_C_API_CHECK(expected_doublet_rate >= 0 && expected_doublet_rate <= 1,
                   SCL_ERROR_INVALID_ARGUMENT, "Expected doublet rate must be in [0, 1]");
    
    SCL_C_API_TRY
        Array<const Real> scores_arr(
            reinterpret_cast<const Real*>(scores),
            n_scores
        );
        
        *threshold_out = static_cast<scl_real_t>(
            scl::kernel::doublet::estimate_threshold(scores_arr, static_cast<Real>(expected_doublet_rate))
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_doublet_detect_bimodal_threshold(
    const scl_real_t* scores,
    const scl_size_t n_scores,
    const scl_index_t n_bins,
    scl_real_t* threshold_out) {
    
    SCL_C_API_CHECK_NULL(scores, "Scores array is null");
    SCL_C_API_CHECK_NULL(threshold_out, "Output threshold pointer is null");
    SCL_C_API_CHECK(n_scores > 0 && n_bins > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<const Real> scores_arr(
            reinterpret_cast<const Real*>(scores),
            n_scores
        );
        
        *threshold_out = static_cast<scl_real_t>(
            scl::kernel::doublet::detect_bimodal_threshold(scores_arr, n_bins)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Doublet Calling
// =============================================================================

SCL_EXPORT scl_error_t scl_doublet_call_doublets(
    const scl_real_t* scores,
    const scl_size_t n_scores,
    const scl_real_t threshold,
    int* is_doublet,
    scl_index_t* n_doublets_out) {
    
    SCL_C_API_CHECK_NULL(scores, "Scores array is null");
    SCL_C_API_CHECK_NULL(is_doublet, "Output is_doublet array is null");
    SCL_C_API_CHECK_NULL(n_doublets_out, "Output n_doublets pointer is null");
    SCL_C_API_CHECK(n_scores > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of scores must be positive");
    
    SCL_C_API_TRY
        Array<const Real> scores_arr(
            reinterpret_cast<const Real*>(scores),
            n_scores
        );
        Array<bool> is_doublet_arr(
            reinterpret_cast<bool*>(is_doublet),
            n_scores
        );
        
        *n_doublets_out = scl::kernel::doublet::call_doublets(
            scores_arr, static_cast<Real>(threshold), is_doublet_arr
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Full Detection Pipeline
// =============================================================================

SCL_EXPORT scl_error_t scl_doublet_detect_doublets(
    scl_sparse_t X,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    scl_real_t* scores,
    int* is_doublet,
    const scl_doublet_method_t method,
    const scl_real_t expected_rate,
    const scl_index_t k_neighbors,
    const uint64_t seed,
    scl_index_t* n_doublets_out) {
    
    SCL_C_API_CHECK_NULL(X, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(scores, "Output scores array is null");
    SCL_C_API_CHECK_NULL(is_doublet, "Output is_doublet array is null");
    SCL_C_API_CHECK_NULL(n_doublets_out, "Output n_doublets pointer is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0 && k_neighbors > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    SCL_C_API_CHECK(expected_rate >= 0 && expected_rate <= 1,
                   SCL_ERROR_INVALID_ARGUMENT, "Expected rate must be in [0, 1]");
    
    SCL_C_API_TRY
        Array<Real> scores_arr(
            reinterpret_cast<Real*>(scores),
            static_cast<Size>(n_cells)
        );
        Array<bool> is_doublet_arr(
            reinterpret_cast<bool*>(is_doublet),
            static_cast<Size>(n_cells)
        );
        
        Index n_doublets = 0;
        
        X->visit([&](auto& mat) {
            n_doublets = scl::kernel::doublet::detect_doublets(
                mat, n_cells, n_genes,
                scores_arr, is_doublet_arr,
                convert_doublet_method(method),
                expected_rate, k_neighbors, seed
            );
        });
        
        *n_doublets_out = n_doublets;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Utility Functions
// =============================================================================

SCL_EXPORT scl_error_t scl_doublet_get_singlet_indices(
    const int* is_doublet,
    const scl_size_t n_cells,
    scl_index_t* singlet_indices,
    const scl_size_t max_indices,
    scl_index_t* n_singlets_out) {
    
    SCL_C_API_CHECK_NULL(is_doublet, "Is doublet array is null");
    SCL_C_API_CHECK_NULL(singlet_indices, "Output singlet indices array is null");
    SCL_C_API_CHECK_NULL(n_singlets_out, "Output n_singlets pointer is null");
    SCL_C_API_CHECK(n_cells > 0 && max_indices > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<const bool> is_doublet_arr(
            reinterpret_cast<const bool*>(is_doublet),
            n_cells
        );
        Array<Index> singlet_arr(
            reinterpret_cast<Index*>(singlet_indices),
            max_indices
        );
        
        *n_singlets_out = scl::kernel::doublet::get_singlet_indices(
            is_doublet_arr, singlet_arr
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_doublet_score_stats(
    const scl_real_t* scores,
    const scl_size_t n_scores,
    scl_real_t* mean_out,
    scl_real_t* std_dev_out,
    scl_real_t* median_out) {
    
    SCL_C_API_CHECK_NULL(scores, "Scores array is null");
    SCL_C_API_CHECK_NULL(mean_out, "Output mean pointer is null");
    SCL_C_API_CHECK_NULL(std_dev_out, "Output std dev pointer is null");
    SCL_C_API_CHECK_NULL(median_out, "Output median pointer is null");
    SCL_C_API_CHECK(n_scores > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of scores must be positive");
    
    SCL_C_API_TRY
        Array<const Real> scores_arr(
            reinterpret_cast<const Real*>(scores),
            n_scores
        );
        
        Real mean = Real(0);
        Real std_dev = Real(0);
        Real median = Real(0);
        
        scl::kernel::doublet::doublet_score_stats(
            scores_arr, &mean, &std_dev, &median
        );
        
        *mean_out = static_cast<scl_real_t>(mean);
        *std_dev_out = static_cast<scl_real_t>(std_dev);
        *median_out = static_cast<scl_real_t>(median);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
