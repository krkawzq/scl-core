// =============================================================================
// FILE: scl/binding/c_api/impute/impute.cpp
// BRIEF: C API implementation for imputation
// =============================================================================

#include "scl/binding/c_api/impute/impute.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/impute.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

scl_error_t scl_impute_knn(
    scl_sparse_t matrix,
    const scl_index_t* knn_indices,
    const scl_real_t* knn_distances,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t k_neighbors,
    scl_real_t* X_imputed,
    scl_real_t bandwidth,
    scl_real_t threshold)
{
    if (!matrix || !knn_indices || !knn_distances || !X_imputed) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        matrix->visit([&](auto& m) {
            scl::kernel::impute::knn_impute_dense(
                m,
                reinterpret_cast<const Index*>(knn_indices),
                reinterpret_cast<const Real*>(knn_distances),
                n_cells,
                n_genes,
                k_neighbors,
                reinterpret_cast<Real*>(X_imputed),
                static_cast<Real>(bandwidth),
                static_cast<Real>(threshold)
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_impute_knn_weighted(
    scl_sparse_t matrix,
    const scl_index_t* knn_indices,
    const scl_real_t* knn_distances,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t k_neighbors,
    scl_real_t* X_imputed,
    scl_real_t bandwidth,
    scl_real_t threshold)
{
    if (!matrix || !knn_indices || !knn_distances || !X_imputed) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        matrix->visit([&](auto& m) {
            scl::kernel::impute::knn_impute_weighted_dense(
                m,
                reinterpret_cast<const Index*>(knn_indices),
                reinterpret_cast<const Real*>(knn_distances),
                n_cells,
                n_genes,
                k_neighbors,
                reinterpret_cast<Real*>(X_imputed),
                static_cast<Real>(bandwidth),
                static_cast<Real>(threshold)
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_impute_diffusion(
    scl_sparse_t matrix,
    scl_sparse_t transition_matrix,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_steps,
    scl_real_t* X_imputed)
{
    if (!matrix || !transition_matrix || !X_imputed) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid() || !transition_matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        // Use sparse transition matrix version
        matrix->visit([&](auto& m) {
            transition_matrix->visit([&](auto& trans) {
                scl::kernel::impute::diffusion_impute_sparse_transition(
                    m,
                    trans,
                    n_cells,
                    n_genes,
                    n_steps,
                    reinterpret_cast<Real*>(X_imputed)
                );
            });
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_impute_magic(
    scl_sparse_t matrix,
    scl_sparse_t affinity_matrix,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t t,
    scl_real_t* X_imputed)
{
    if (!matrix || !affinity_matrix || !X_imputed) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid() || !affinity_matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        matrix->visit([&](auto& m) {
            affinity_matrix->visit([&](auto& aff) {
                scl::kernel::impute::magic_impute(
                    m,
                    aff,
                    n_cells,
                    n_genes,
                    t,
                    reinterpret_cast<Real*>(X_imputed)
                );
            });
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

