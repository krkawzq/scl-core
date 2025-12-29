// =============================================================================
// FILE: scl/binding/c_api/neighbors.cpp
// BRIEF: C API implementation for K-nearest neighbors
// =============================================================================

#include "scl/binding/c_api/neighbors.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/neighbors.hpp"
#include "scl/core/type.hpp"
#include "scl/core/memory.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Compute Norms
// =============================================================================

SCL_EXPORT scl_error_t scl_neighbors_compute_norms(
    scl_sparse_t matrix,
    scl_real_t* norms_sq) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(norms_sq, "Norms output pointer is null");

    SCL_C_API_TRY
        const Index n = matrix->rows();
        const Size n_sz = static_cast<Size>(n);
        Array<Real> norms_arr(reinterpret_cast<Real*>(norms_sq), n_sz);
        
        matrix->visit([&](auto& m) {
            scl::kernel::neighbors::compute_norms(m, norms_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// K-Nearest Neighbors
// =============================================================================

SCL_EXPORT scl_error_t scl_knn(
    scl_sparse_t matrix,
    const scl_real_t* norms_sq,
    const scl_size_t n_samples,
    const scl_size_t k,
    scl_index_t* out_indices,
    scl_real_t* out_distances) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(norms_sq, "Norms input pointer is null");
    SCL_C_API_CHECK_NULL(out_indices, "Indices output pointer is null");
    SCL_C_API_CHECK_NULL(out_distances, "Distances output pointer is null");
    SCL_C_API_CHECK(n_samples > 0 && k > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");

    SCL_C_API_TRY
        const Size n_samples_sz = static_cast<Size>(n_samples);
        const Size k_sz = static_cast<Size>(k);
        const Size total_neighbors = n_samples_sz * k_sz;
        
        Array<const Real> norms_arr(reinterpret_cast<const Real*>(norms_sq), n_samples_sz);
        Array<Index> indices_arr(reinterpret_cast<Index*>(out_indices), total_neighbors);
        Array<Real> distances_arr(reinterpret_cast<Real*>(out_distances), total_neighbors);
        
        matrix->visit([&](auto& m) {
            scl::kernel::neighbors::knn(m, norms_arr, k_sz, indices_arr, distances_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// K-Nearest Neighbors (Simple - computes norms internally)
// =============================================================================

scl_error_t scl_knn_simple(
    scl_sparse_t matrix,
    scl_size_t n_samples,
    scl_size_t k,
    scl_index_t* out_indices,
    scl_real_t* out_distances)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(out_indices, "Indices output pointer is null");
    SCL_C_API_CHECK_NULL(out_distances, "Distances output pointer is null");

    SCL_C_API_TRY {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        SCL_C_API_CHECK(wrapper->valid(), SCL_ERROR_INVALID_ARGUMENT,
                       "Invalid sparse matrix");

        const Size n_samples_sz = static_cast<Size>(n_samples);
        const Size k_sz = static_cast<Size>(k);
        const Size total_neighbors = n_samples_sz * k_sz;
        
        // Compute norms first
        auto norms_sq_ptr = scl::memory::aligned_alloc<Real>(n_samples_sz, SCL_ALIGNMENT);
        Real* norms_sq = norms_sq_ptr.get();
        
        Array<Real> norms_arr(norms_sq, n_samples_sz);
        matrix->visit([&](auto& m) {
            scl::kernel::neighbors::compute_norms(m, norms_arr);
        });
        
        // Compute KNN
        Array<const Real> norms_const_arr(norms_sq, n_samples_sz);
        Array<Index> indices_arr(reinterpret_cast<Index*>(out_indices), total_neighbors);
        Array<Real> distances_arr(reinterpret_cast<Real*>(out_distances), total_neighbors);
        
        matrix->visit([&](auto& m) {
            scl::kernel::neighbors::knn(m, norms_const_arr, k_sz, indices_arr, distances_arr);
        });
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

} // extern "C"
