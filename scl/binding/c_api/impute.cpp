// =============================================================================
// FILE: scl/binding/c_api/impute/impute.cpp
// BRIEF: C API implementation for imputation
// =============================================================================

#include "scl/binding/c_api/impute.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/impute.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// KNN Imputation
// =============================================================================

SCL_EXPORT scl_error_t scl_impute_knn(
    scl_sparse_t matrix,
    const scl_index_t* knn_indices,
    const scl_real_t* knn_distances,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_index_t k_neighbors,
    scl_real_t* X_imputed,
    const scl_real_t bandwidth,
    const scl_real_t threshold) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(knn_indices, "KNN indices array is null");
    SCL_C_API_CHECK_NULL(knn_distances, "KNN distances array is null");
    SCL_C_API_CHECK_NULL(X_imputed, "Output imputed matrix is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0 && k_neighbors > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        matrix->visit([&](auto& m) {
            scl::kernel::impute::knn_impute_dense(
                m,
                reinterpret_cast<const Index*>(knn_indices),
                reinterpret_cast<const Real*>(knn_distances),
                n_cells, n_genes, k_neighbors,
                reinterpret_cast<Real*>(X_imputed),
                static_cast<Real>(bandwidth),
                static_cast<Real>(threshold)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_impute_knn_weighted(
    scl_sparse_t matrix,
    const scl_index_t* knn_indices,
    const scl_real_t* knn_distances,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_index_t k_neighbors,
    scl_real_t* X_imputed,
    const scl_real_t bandwidth,
    const scl_real_t threshold) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(knn_indices, "KNN indices array is null");
    SCL_C_API_CHECK_NULL(knn_distances, "KNN distances array is null");
    SCL_C_API_CHECK_NULL(X_imputed, "Output imputed matrix is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0 && k_neighbors > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        matrix->visit([&](auto& m) {
            scl::kernel::impute::knn_impute_dense(
                m,
                reinterpret_cast<const Index*>(knn_indices),
                reinterpret_cast<const Real*>(knn_distances),
                n_cells, n_genes, k_neighbors,
                reinterpret_cast<Real*>(X_imputed),
                static_cast<Real>(bandwidth),
                static_cast<Real>(threshold)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Diffusion Imputation
// =============================================================================

SCL_EXPORT scl_error_t scl_impute_diffusion(
    scl_sparse_t matrix,
    scl_sparse_t transition_matrix,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_index_t n_steps,
    scl_real_t* X_imputed) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(transition_matrix, "Transition matrix is null");
    SCL_C_API_CHECK_NULL(X_imputed, "Output imputed matrix is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0 && n_steps > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Size total = static_cast<Size>(n_cells) * static_cast<Size>(n_genes);
        Array<Real> X_arr(reinterpret_cast<Real*>(X_imputed), total);
        
        matrix->visit([&](auto& m) {
            transition_matrix->visit([&](auto& trans) {
                scl::kernel::impute::diffusion_impute_sparse_transition(
                    m, trans, n_cells, n_genes, n_steps,
                    reinterpret_cast<Real*>(X_imputed)
                );
            });
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// MAGIC Imputation
// =============================================================================

SCL_EXPORT scl_error_t scl_impute_magic(
    scl_sparse_t matrix,
    scl_sparse_t affinity_matrix,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_index_t t,
    scl_real_t* X_imputed) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(affinity_matrix, "Affinity matrix is null");
    SCL_C_API_CHECK_NULL(X_imputed, "Output imputed matrix is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0 && t > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Size total = static_cast<Size>(n_cells) * static_cast<Size>(n_genes);
        Array<Real> X_arr(reinterpret_cast<Real*>(X_imputed), total);
        
        matrix->visit([&](auto& m) {
            affinity_matrix->visit([&](auto& aff) {
                scl::kernel::impute::magic_impute(
                    m, aff, n_cells, n_genes, t,
                    reinterpret_cast<Real*>(X_imputed)
                );
            });
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
